"""Dropout-LayerNorm Correction (DLC).

Inverted dropout is unbiased per-channel at eval time (E[dropout(x)] = x),
but LayerNorm's normalization statistics (mean/std) are shared across
channels, so E[LayerNorm(dropout(x))] != LayerNorm(x). Every block of
StructureModule contains this pattern twice (the post-IPA LayerNorm and
the transition's own internal LayerNorm), both fed directly by a
dropout layer with no intervening operation. This introduces a small
systematic bias into eval-mode structure predictions.

This module applies the closed-form correction derived in
"Correcting the Dropout-LayerNorm Expectation Gap Improves Protein
Structure Models" (https://arxiv.org/abs/2609.32062), which empirically
improved FlashABB's whole-Fv RMSD by 10.9% in that paper's headline test.
See install(...) below for usage; it is applied automatically by
`pretrained(dlc_correction=True)` (the default).
"""
import torch

from .structure_transformer import StructureModule

DROPOUT_KEEP_PROB = 0.9  # q = 1 - dropout_rate (0.1) used throughout StructureModule


def _theory_c(mu: torch.Tensor, sigma: torch.Tensor, q: float) -> torch.Tensor:
    """Correction factor for a dropout->LayerNorm site with no residual
    bypass. mu, sigma are the pre-dropout activation's per-position
    channel-wise mean/std (last-dim reductions, keepdim). q is the
    dropout keep probability (1 - drop_rate)."""
    p = 1.0 - q
    return torch.sqrt(q / (1.0 + p * (mu / sigma) ** 2))


def _apply_correction(ln_output: torch.Tensor, ln_bias: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Applies the correction to a LayerNorm's already-computed output,
    given its bias parameter. Only the gamma*norm(x) part is shrunk; the
    bias is recovered at full strength (exact algebraically)."""
    return c * ln_output + (1.0 - c) * ln_bias


def correction_sites(model: StructureModule):
    """Returns the list of (dropout_module, layernorm_module) site pairs
    in a StructureModule: the shared IPA dropout site (paired with each
    block's own layer_norm_ipa_layers[i]) and each block's own
    transition dropout/layer_norm."""
    sites = [(model.ipa_dropout, model.layer_norm_ipa_layers[i]) for i in range(model.no_blocks)]
    sites += [(model.transition_layers[i].dropout, model.transition_layers[i].layer_norm) for i in range(model.no_blocks)]
    return sites


def install(model: StructureModule, mode_flag: dict, q: float = DROPOUT_KEEP_PROB):
    """Registers forward hooks on every dropout->LayerNorm site in
    `model` (a StructureModule) that apply the DLC correction.
    `mode_flag` is a mutable dict with a "value" key: "off" leaves the
    forward pass untouched, any other value ("on") applies the
    correction. Hooks are installed once and gated by `mode_flag`
    thereafter, so DLC can be toggled at runtime without reconstructing
    the model (see `pretrained.dlc_correction`).

    Idempotent guard: does nothing if hooks are already installed on
    this model (checked via `model._dlc_hooks_installed`).
    """
    if getattr(model, "_dlc_hooks_installed", False):
        return
    model._dlc_hooks_installed = True

    for dropout_module, ln_module in correction_sites(model):
        live_c = {"value": None}

        def capture_hook(module, args, live_c=live_c):
            if mode_flag["value"] == "off":
                return
            x = args[0]
            mu = x.mean(dim=-1, keepdim=True)
            sigma = x.std(dim=-1, keepdim=True, unbiased=False)
            live_c["value"] = _theory_c(mu, sigma, q)

        def correction_hook(module, inp, out, live_c=live_c):
            if mode_flag["value"] == "off":
                return out
            return _apply_correction(out, module.bias, live_c["value"])

        dropout_module.register_forward_pre_hook(capture_hook)
        ln_module.register_forward_hook(correction_hook)
