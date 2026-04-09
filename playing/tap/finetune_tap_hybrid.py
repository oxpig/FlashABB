"""Train hybrid TAP regressor using concatenated FlashABB + AbLang2 embeddings."""

import os
import sys
import copy
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

TAP_COLS = ["PSH", "PPC", "PNC", "SFvCSP"]
SEED = 42
DIR = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# Encoders (copied from finetune_tap.py)
# ---------------------------------------------------------------------------

class FlashABBEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        from flash_abb.load_model import load_model
        self.flabb, _ = load_model("flash-abb")
        self.flabb.to(device)
        self._device = device
        self.embed_dim = 128

    def forward(self, seqs):
        """Returns per-residue embeddings and mask."""
        from flash_abb.model.flash_abb import featurize
        features = featurize(seqs, self._device)
        output = self.flabb.model(
            {"single": features["single"]},
            features["aatype"],
            features["res_idx"],
            features["mask"],
        )
        single = output["single"]  # (batch, seq_len, 128)
        mask = features["mask"]  # (batch, seq_len)
        return single, mask

    def load_state_dict(self, state_dict):
        self.flabb.load_state_dict(state_dict)

    def state_dict(self):
        return self.flabb.state_dict()

    def parameters(self):
        return self.flabb.parameters()


class AbLang2Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        import ablang2
        self.ablang = ablang2.pretrained("ablang2-paired", device=device)
        self.device = device
        self.embed_dim = 480

    def forward(self, seqs):
        """Returns per-residue embeddings and mask, with separator removed to align with FlashABB."""
        tokenized = self.ablang.tokenizer(
            seqs, pad=True, w_extra_tkns=False, device=self.device
        )
        with torch.no_grad():
            rescoding = self.ablang.AbRep(tokenized).last_hidden_states  # (batch, seq_len, 480)

        # AbLang2 includes separator token (25 = "|"), FlashABB doesn't
        # Remove separator to align sequences
        sep_token_id = 25
        is_sep = tokenized == sep_token_id  # (B, L)

        batch_size, seq_len, embed_dim = rescoding.shape
        embeddings_no_sep = []
        masks_no_sep = []

        for i in range(batch_size):
            non_sep_mask = ~is_sep[i]
            emb = rescoding[i][non_sep_mask]
            embeddings_no_sep.append(emb)
            mask = tokenized[i][non_sep_mask] != 0
            masks_no_sep.append(mask)

        # Pad to same length
        max_len = max(e.size(0) for e in embeddings_no_sep)
        rescoding_aligned = torch.zeros(batch_size, max_len, embed_dim, device=self.device)
        mask_aligned = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)

        for i, (emb, mask) in enumerate(zip(embeddings_no_sep, masks_no_sep)):
            length = emb.size(0)
            rescoding_aligned[i, :length] = emb
            mask_aligned[i, :length] = mask

        return rescoding_aligned, mask_aligned

    def load_state_dict(self, state_dict):
        self.ablang.AbRep.load_state_dict(state_dict)

    def state_dict(self):
        return self.ablang.AbRep.state_dict()

    def parameters(self):
        return self.ablang.AbRep.parameters()


# ---------------------------------------------------------------------------
# Hybrid Model
# ---------------------------------------------------------------------------

class HybridTAPRegressor(nn.Module):
    """Concatenates per-residue embeddings, applies MLP, then sum pools."""
    def __init__(self, flabb_dim=128, ablang_dim=480, hidden_dim=256):
        super().__init__()
        combined_dim = flabb_dim + ablang_dim  # 608

        # Per-residue fusion MLP that outputs TAP properties
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4),  # Output 4 TAP properties per residue
        )

    def forward(self, flabb_emb, ablang_emb, mask):
        """
        Args:
            flabb_emb: (batch, seq_len, 128) per-residue FlashABB embeddings
            ablang_emb: (batch, seq_len, 480) per-residue AbLang2 embeddings (separator removed)
            mask: (batch, seq_len) padding mask from FlashABB
        Returns:
            (batch, 4) predictions
        """
        # Both sequences should now be aligned (separator removed from AbLang2)
        # Concatenate per-residue embeddings
        combined = torch.cat([flabb_emb, ablang_emb], dim=-1)  # (batch, seq_len, 608)

        # Apply fusion MLP to each residue to get per-residue TAP predictions
        per_residue_tap = self.fusion_mlp(combined)  # (batch, seq_len, 4)

        # Masked sum pooling - sum TAP contributions from all residues
        mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        masked_tap = per_residue_tap * mask_expanded  # (batch, seq_len, 4)
        summed_tap = masked_tap.sum(dim=1)  # (batch, 4)

        return summed_tap


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SeqDataset(Dataset):
    def __init__(self, seqs, targets):
        self.seqs = seqs
        self.targets = targets

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.targets[idx]


def seq_collate(batch):
    seqs, targets = zip(*batch)
    return list(seqs), torch.stack(targets)


# ---------------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------------

def load_data():
    import pandas as pd
    csv_path = os.path.join(DIR, "OAS_paired_with_tap.csv")
    df = pd.read_csv(csv_path)
    seqs = [s.replace("/", "|") for s in df["full_seq"].tolist()]
    targets = torch.tensor(df[TAP_COLS].values, dtype=torch.float32)
    return seqs, targets


def split_indices(n, seed=SEED):
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_hybrid(
    flabb_encoder,
    ablang_encoder,
    head,
    train_set,
    val_set,
    tgt_mean,
    tgt_std,
    flabb_lr=1e-5,
    ablang_lr=1e-5,
    head_lr=1e-3,
    epochs=50,
    patience=10,
    batch_size=16,
    device="cuda",
    use_wandb=False,
):
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=seq_collate
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, collate_fn=seq_collate
    )

    tgt_mean_d = tgt_mean.to(device)
    tgt_std_d = tgt_std.to(device)

    # Separate optimizers for each encoder + head
    optimizer = torch.optim.Adam([
        {"params": flabb_encoder.parameters(), "lr": flabb_lr},
        {"params": ablang_encoder.parameters(), "lr": ablang_lr},
        {"params": head.parameters(), "lr": head_lr},
    ])
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_flabb_state = None
    best_ablang_state = None
    best_head_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        flabb_encoder.train()
        ablang_encoder.train()
        head.train()
        train_loss = 0.0
        n_train = 0
        n_batches = len(train_loader)

        for batch_idx, (seqs, targets) in enumerate(train_loader, 1):
            targets_norm = (targets.to(device) - tgt_mean_d) / tgt_std_d

            # Get per-residue embeddings from both encoders
            flabb_emb, flabb_mask = flabb_encoder(seqs)
            ablang_emb, ablang_mask = ablang_encoder(seqs)

            # Use FlashABB mask (more reliable for antibody sequences)
            mask = flabb_mask.float()

            # Hybrid prediction
            pred = head(flabb_emb, ablang_emb, mask)
            loss = criterion(pred, targets_norm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            train_loss += batch_loss * len(seqs)
            n_train += len(seqs)
            print(f"\r  Epoch {epoch:3d}  batch {batch_idx}/{n_batches}  loss={batch_loss:.4f}", end="", flush=True)

        print()
        train_loss /= n_train

        # Validation
        flabb_encoder.eval()
        ablang_encoder.eval()
        head.eval()
        val_preds, val_actuals = [], []

        with torch.no_grad():
            for seqs, targets in val_loader:
                flabb_emb, flabb_mask = flabb_encoder(seqs)
                ablang_emb, ablang_mask = ablang_encoder(seqs)
                mask = flabb_mask.float()
                pred_norm = head(flabb_emb, ablang_emb, mask)
                pred = pred_norm * tgt_std_d + tgt_mean_d
                val_preds.append(pred.cpu())
                val_actuals.append(targets)

        val_preds = torch.cat(val_preds)
        val_actuals = torch.cat(val_actuals)

        # Validation loss (normalized)
        val_loss = criterion(
            (val_preds - tgt_mean) / tgt_std,
            (val_actuals - tgt_mean) / tgt_std,
        ).item()

        # Per-property MAE & R²
        maes, r2s = [], []
        for i in range(len(TAP_COLS)):
            p, a = val_preds[:, i], val_actuals[:, i]
            maes.append((p - a).abs().mean().item())
            ss_res = ((a - p) ** 2).sum().item()
            ss_tot = ((a - a.mean()) ** 2).sum().item()
            r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))

        mae_str = "  ".join(f"{c}: {m:.3f}" for c, m in zip(TAP_COLS, maes))
        r2_str = "  ".join(f"{c}: {r:.3f}" for c, r in zip(TAP_COLS, r2s))
        star = " *" if val_loss < best_val_loss else ""
        print(
            f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{star}\n"
            f"    MAE  {mae_str}\n"
            f"    R²   {r2_str}"
        )

        if use_wandb:
            import wandb
            log = {"train/loss": train_loss, "val/loss": val_loss}
            for col, mae, r2 in zip(TAP_COLS, maes, r2s):
                log[f"val/MAE_{col}"] = mae
                log[f"val/R2_{col}"] = r2
            wandb.log(log, step=epoch)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_flabb_state = copy.deepcopy(flabb_encoder.state_dict())
            best_ablang_state = copy.deepcopy(ablang_encoder.state_dict())
            best_head_state = copy.deepcopy(head.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best states
    flabb_encoder.load_state_dict(best_flabb_state)
    ablang_encoder.load_state_dict(best_ablang_state)
    head.load_state_dict(best_head_state)

    return flabb_encoder, ablang_encoder, head


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(flabb_encoder, ablang_encoder, head, test_set, tgt_mean, tgt_std, batch_size=16, device="cuda"):
    loader = DataLoader(test_set, batch_size=batch_size, collate_fn=seq_collate)
    tgt_mean_d = tgt_mean.to(device)
    tgt_std_d = tgt_std.to(device)

    flabb_encoder.eval()
    ablang_encoder.eval()
    head.eval()

    all_pred, all_actual = [], []
    with torch.no_grad():
        for seqs, targets in loader:
            flabb_emb, flabb_mask = flabb_encoder(seqs)
            ablang_emb, ablang_mask = ablang_encoder(seqs)
            mask = flabb_mask.float()
            pred_norm = head(flabb_emb, ablang_emb, mask)
            pred = pred_norm * tgt_std_d + tgt_mean_d
            all_pred.append(pred.cpu())
            all_actual.append(targets)

    pred = torch.cat(all_pred)
    actual = torch.cat(all_actual)

    results = {}
    for i, col in enumerate(TAP_COLS):
        p, a = pred[:, i], actual[:, i]
        mse = ((p - a) ** 2).mean().item()
        mae = (p - a).abs().mean().item()
        ss_res = ((a - p) ** 2).sum().item()
        ss_tot = ((a - a.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        results[col] = {"MSE": mse, "MAE": mae, "R2": r2}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--flabb_lr", type=float, default=1e-5)
    parser.add_argument("--ablang_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  Hybrid FlashABB + AbLang2 TAP Regression")
    print("="*70)

    if args.wandb:
        import wandb
        wandb.init(
            project="tap-regression",
            name="Hybrid-FlashABB-AbLang2",
            config=vars(args),
        )

    # Load data
    print("\nLoading data...")
    seqs, targets = load_data()
    train_idx, val_idx, test_idx = split_indices(len(seqs))

    train_seqs = [seqs[i] for i in train_idx]
    val_seqs = [seqs[i] for i in val_idx]
    test_seqs = [seqs[i] for i in test_idx]

    train_targets = targets[train_idx]
    val_targets = targets[val_idx]
    test_targets = targets[test_idx]

    # Normalize targets
    tgt_mean = train_targets.mean(dim=0)
    tgt_std = train_targets.std(dim=0)

    train_set = SeqDataset(train_seqs, train_targets)
    val_set = SeqDataset(val_seqs, val_targets)
    test_set = SeqDataset(test_seqs, test_targets)

    print(f"  Train: {len(train_set)}")
    print(f"  Val:   {len(val_set)}")
    print(f"  Test:  {len(test_set)}")

    # Create encoders and head
    print("\nInitializing encoders...")
    flabb_encoder = FlashABBEncoder(args.device)
    ablang_encoder = AbLang2Encoder(args.device)
    head = HybridTAPRegressor(flabb_dim=128, ablang_dim=480, hidden_dim=256).to(args.device)

    print(f"  FlashABB embedding dim: {flabb_encoder.embed_dim}")
    print(f"  AbLang2 embedding dim:  {ablang_encoder.embed_dim}")
    print(f"  Combined dim: {flabb_encoder.embed_dim + ablang_encoder.embed_dim}")

    # Train
    print("\nTraining...")
    flabb_encoder, ablang_encoder, head = train_hybrid(
        flabb_encoder,
        ablang_encoder,
        head,
        train_set,
        val_set,
        tgt_mean,
        tgt_std,
        flabb_lr=args.flabb_lr,
        ablang_lr=args.ablang_lr,
        head_lr=args.head_lr,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
        use_wandb=args.wandb,
    )

    # Save checkpoint
    ckpt_path = os.path.join(DIR, "tap_ft_hybrid.pt")
    torch.save(
        {
            "flabb_encoder_state": flabb_encoder.state_dict(),
            "ablang_encoder_state": ablang_encoder.state_dict(),
            "head_state": head.state_dict(),
            "tgt_mean": tgt_mean,
            "tgt_std": tgt_std,
        },
        ckpt_path,
    )
    print(f"\nSaved checkpoint to {ckpt_path}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate(flabb_encoder, ablang_encoder, head, test_set, tgt_mean, tgt_std,
                      batch_size=args.batch_size, device=args.device)

    print("\nTest Results:")
    print("-" * 70)
    print(f"{'Property':<12} {'MSE':>12} {'MAE':>12} {'R²':>12}")
    print("-" * 70)
    for col in TAP_COLS:
        r = results[col]
        print(f"{col:<12} {r['MSE']:>12.4f} {r['MAE']:>12.4f} {r['R2']:>12.4f}")

    if args.wandb:
        import wandb
        test_log = {}
        for col, r in results.items():
            test_log[f"test/MSE_{col}"] = r["MSE"]
            test_log[f"test/MAE_{col}"] = r["MAE"]
            test_log[f"test/R2_{col}"] = r["R2"]
        wandb.log(test_log)
        wandb.finish()


if __name__ == "__main__":
    main()
