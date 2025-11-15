import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from ..neural_network import Anon
from .pgn_dataset import ChessIterableDataset, PGNDataset
import argparse


def tiny_supervised_train(pgn_path: str = None,
                          batch_size: int = 128,
                          epochs: int = 10,
                          lr: float = 3e-4,
                          num_workers: int = 4,
                          precompute: bool = True,
                          cache_path: str = None,
                          val_split: float = 0.05,
                          checkpoint_dir: str = "checkpoints",
                          save_every: int = 1):
    """Train a model on PGN data with sensible defaults for larger runs.

    Uses GPU + AMP when available, DataLoader optimizations, gradient clipping,
    and a simple learning-rate scheduler.
    """
    if pgn_path is None:
        print("No PGN provided. Use --pgn <file> to train.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PGNDataset(pgn_path, max_games=5000, max_moves_per_game=200, precompute=precompute, cache_path=cache_path)
    if len(ds) == 0:
        print("Dataset empty. Check PGN or use --demo")
        return

    # Create checkpoint dir
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Split into train/val
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

    train_loader = DataLoader(Subset(ds, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = None
    if len(val_indices) > 0:
        val_loader = DataLoader(Subset(ds, val_indices), batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers//2), pin_memory=pin_memory)

    model = Anon().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            positions, targets, outcomes = batch

            positions = positions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outcomes = outcomes.to(device, non_blocking=True)

            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, values = model(positions)
                loss_policy = F.cross_entropy(logits, targets)
                loss_value = F.mse_loss(values, outcomes)
                loss = loss_policy + loss_value

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()
            if i % 50 == 0:
                print(f"epoch {epoch} iter {i} loss {loss.item():.4f} p:{loss_policy.item():.4f} v:{loss_value.item():.4f}")

        scheduler.step()
        avg_train = running_loss / max(1, len(train_loader))

        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for vbatch in val_loader:
                    vpos, vtgt, vout = vbatch
                    vpos = vpos.to(device, non_blocking=True)
                    vtgt = vtgt.to(device, non_blocking=True)
                    vout = vout.to(device, non_blocking=True)
                    vlogits, vvals = model(vpos)
                    vloss_p = F.cross_entropy(vlogits, vtgt)
                    vloss_v = F.mse_loss(vvals, vout)
                    val_running += (vloss_p + vloss_v).item()
            val_loss = val_running / max(1, len(val_loader))
            print(f"Epoch {epoch} completed. Train loss: {avg_train:.4f} Val loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch} completed. Train loss: {avg_train:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"model_epoch_{epoch+1}.pt"
            torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'opt_state': opt.state_dict()}, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # Save best
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            best_path = ckpt_dir / "best_model.pt"
            torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'opt_state': opt.state_dict(), 'val_loss': val_loss}, best_path)
            print(f"New best model saved: {best_path} (val_loss={val_loss:.4f})")

    print("Done training")


def main():
    parser = argparse.ArgumentParser(description="Train a chess neural network from PGN files")
    parser.add_argument("--pgn", required=True, help="Path to PGN file or directory containing PGNs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--no-precompute", dest="precompute", action="store_false", help="Disable precomputing dataset tensors")
    parser.add_argument("--cache-path", type=str, default=None, help="Optional cache path for precomputed tensors (.pt)")
    parser.add_argument("--val-split", type=float, default=0.05, help="Fraction of dataset to hold out for validation")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")

    args = parser.parse_args()

    print(f"Starting training with PGN={args.pgn} epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}")
    tiny_supervised_train(pgn_path=args.pgn,
                          batch_size=args.batch_size,
                          epochs=args.epochs,
                          lr=args.lr,
                          num_workers=args.num_workers,
                          precompute=args.precompute,
                          cache_path=args.cache_path,
                          val_split=args.val_split,
                          checkpoint_dir=args.checkpoint_dir,
                          save_every=args.save_every)


if __name__ == "__main__":
    main()
