import os, subprocess, json, argparse,requests
from yaml import load, Loader
import torch

list_of_models = {
    "flash-abb":"flabb_weights.pt",
    "flash-abb_masked":"flabb_masked_weights.pt",
}
flash_abb_models = ["flash-abb", "flash-abb_masked"]


def load_model(model_to_use="flash-abb", random_init=False, device='cpu'):

    if model_to_use in flash_abb_models:
        flabb, hparams = fetch_flash_abb(
            model_to_use, 
            random_init=random_init, 
            device=device
        )
    else: 
        assert False, f"The selected model to use ({model_to_use}) does not exist.\
        Please select a valid model."   

    return flabb, hparams


def fetch_flash_abb(model_to_use, random_init=False, device='cpu'):

    from .model.flash_abb import FlashABB

    local_model_folder = os.path.join(os.path.dirname(__file__), "weights")
    file_model = list_of_models[model_to_use]

    with open(os.path.join(local_model_folder, 'params.yaml'), 'r', encoding='utf-8') as f:
        hparams = argparse.Namespace(**load(f, Loader=Loader)).model

    flabb = FlashABB(hparams)
    if not random_init:
        ckpt = torch.load(
            os.path.join(local_model_folder, file_model),
            map_location=torch.device(device),
            weights_only=False,
        )
        flabb.load_state_dict(ckpt)

    return flabb, hparams


def fetch_sss(random_init=False, device='cpu'):
    from .model.seq2struct2seq import BERTCoords

    model = BERTCoords(device=device)
    if not random_init:
        weights_path = os.path.join(os.path.dirname(__file__), "weights", "sss_weights.pt")
        ckpt = torch.load(weights_path, map_location=torch.device(device), weights_only=False)
        model.load_state_dict(ckpt)
    return model.to(device)


def fetch_tap(random_init=False, device='cpu'):
    from .model.seq2struct2seq import BERTCoords
    from .model.tap_head import TAPHead

    encoder = BERTCoords(device=device)
    head = TAPHead()
    if not random_init:
        weights_path = os.path.join(os.path.dirname(__file__), "weights", "tap_weights.pt")
        ckpt = torch.load(weights_path, map_location=torch.device(device), weights_only=False)
        # encoder_state keys have a 'model.' prefix from the training wrapper
        encoder_state = {k.removeprefix('model.'): v for k, v in ckpt['encoder_state'].items()}
        encoder.load_state_dict(encoder_state, strict=False)
        head.load_state_dict(ckpt['head_state'], strict=False)
        head.tgt_mean.copy_(ckpt['tgt_mean'])
        head.tgt_std.copy_(ckpt['tgt_std'])
    return encoder.to(device), head.to(device)
