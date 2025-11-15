import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..neural_network import Anon
from .pgn_dataset import PGNDataset


# -------------------------
# Device selection
# -------------------------

def get_device() -> torch.device:
    """
    For Modal: this will typically be CUDA (A100).
    For local CPU runs, falls back to 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------------
# Main training function
# -------------------------

def tiny_supervised_train(
    pgn_path: Optional[str] = None,
    batch_size: int = 128,
    epochs: int = 10,
    lr: float = 3e-4,
    num_workers: int = 4,
    precompute: bool = True,
    cache_path: Optional[str] = None,
    val_split: float = 0.05,
    checkpoint_dir: str = "checkpoints",
    save_every: int = 1,
) -> None:
    """
    Train the Anon model on PGN data.

    - Uses cross-entropy loss for the policy head.
    - Uses MSE loss for the value head.
    - Splits dataset into train/val and tracks best val loss.
    - Supports CUDA + AMP (GradScaler) on GPU.
    - Saves per-epoch checkpoints + best model.

    Parameters
    ----------
    pgn_path : str
        Path to a PGN file (or directory, depending on your PGNDataset).
    batch_size : int
        Training batch size.
    epochs : int
        Number of epochs to train for.
    lr : float
        Learning rate for Adam.
    num_workers : int
        DataLoader worker threads.
    precompute : bool
        Whether PGNDataset should precompute tensors.
    cache_path : str or None
        Optional path for cached dataset tensors.
    val_split : float
        Fraction of dataset to hold out for validation.
    checkpoint_dir : str
        Directory where checkpoints will be saved.
    save_every : int
        Save a full checkpoint every N epochs.
    """
    if pgn_path is None:
        print("[train] No PGN provided. Use --pgn <file> to train.")
        return

    device = get_device()
    print(f"[train] Using device: {device}")

    # -------------------------
    # Dataset + split
    # -------------------------
    ds = PGNDataset(
        pgn_path,
        max_games=50000,
        max_moves_per_game=200,
        precompute=precompute,
        cache_path=cache_path,
    )
    if len(ds) == 0:
        print("[train] Dataset empty. Check PGN / parameters.")
        return

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n = len(ds)
    indices = list(range(n))
    split = int(n * val_split)

    if split > 0:
        val_indices = indices[:split]
        train_indices = indices[split:]
    else:
        val_indices = []
        train_indices = indices

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        Subset(ds, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = None
    if len(val_indices) > 0:
        val_loader = DataLoader(
            Subset(ds, val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            pin_memory=pin_memory,
        )

    # -------------------------
    # Model, optimizer, scheduler
    # -------------------------
    model = Anon().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    # AMP (CUDA only)
    use_amp = device.type == "cuda"
    if use_amp:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler(enabled=True)
        autocast_ctx = autocast
    else:
        scaler = None
        from contextlib import nullcontext
        autocast_ctx = lambda: nullcontext()

    best_val = float("inf")

    # -------------------------
    # Epoch loop
    # -------------------------
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            positions, targets, outcomes = batch

            positions = positions.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True)          # long
            outcomes = outcomes.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)

            with autocast_ctx():
                logits, values = model(positions)

                # Policy loss
                loss_policy = F.cross_entropy(logits, targets)

                # Value loss: ensure shapes match (B,)
                values = values.view_as(outcomes)
                loss_value = F.mse_loss(values, outcomes)

                loss = loss_policy + loss_value

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                opt.step()

            running_loss += loss.item()

            if i % 50 == 0:
                print(
                    f"[train] epoch {epoch} iter {i} "
                    f"loss {loss.item():.4f} p:{loss_policy.item():.4f} v:{loss_value.item():.4f}"
                )

        scheduler.step()
        avg_train = running_loss / max(1, len(train_loader))

        # -------------------------
        # Validation
        # -------------------------
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0

            with torch.no_grad():
                for vpos, vtgt, vout in val_loader:
                    vpos = vpos.to(device, non_blocking=True).float()
                    vtgt = vtgt.to(device, non_blocking=True)
                    vout = vout.to(device, non_blocking=True).float()

                    with autocast_ctx():
                        vlogits, vvals = model(vpos)
                        vvals = vvals.view_as(vout)

                        vloss_p = F.cross_entropy(vlogits, vtgt)
                        vloss_v = F.mse_loss(vvals, vout)
                        vloss = vloss_p + vloss_v

                    val_running += vloss.item()

            val_loss = val_running / max(1, len(val_loader))
            print(
                f"[epoch {epoch}] train_loss={avg_train:.4f}  "
                f"val_loss={val_loss:.4f}"
            )
        else:
            print(f"[epoch {epoch}] train_loss={avg_train:.4f}")

        # -------------------------
        # Checkpointing
        # -------------------------

        # Save checkpoint every N epochs
        if (epoch + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"model_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                },
                ckpt_path,
            )
            print(f"[ckpt] Saved checkpoint: {ckpt_path}")

        # Save best model based on val loss
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            best_path = ckpt_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "val_loss": val_loss,
                },
                best_path,
            )
            print(f"[ckpt] New best model: {best_path} (val_loss={val_loss:.4f})")

    print("[train] Done training.")


# -------------------------
# CLI wrapper (optional)
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a chess neural network from PGN files")
    parser.add_argument("--pgn", required=True, help="Path to PGN file or directory containing PGNs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers")

    parser.add_argument(
        "--no-precompute",
        dest="precompute",
        action="store_false",
        help="Disable precomputing dataset tensors",
    )
    parser.set_defaults(precompute=True)

    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Optional cache path for precomputed tensors (.pt)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Fraction of dataset to hold out for validation",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )

    args = parser.parse_args()

    print(
        f"[cli] Starting training with "
        f"PGN={args.pgn} epochs={args.epochs} "
        f"batch_size={args.batch_size} lr={args.lr}"
    )

    tiny_supervised_train(
        pgn_path=args.pgn,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        precompute=args.precompute,
        cache_path=args.cache_path,
        val_split=args.val_split,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
