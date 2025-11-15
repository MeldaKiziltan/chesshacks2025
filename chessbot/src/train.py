import argparse
import modal

# Modal setup: persisted volume for PGNs / checkpoints
stub = modal.Stub(name="chess-training-modal")
data_vol = modal.Volume.persisted("chess-data-vol")


@stub.function(
    image=modal.Image.debian_slim().pip_install("torch", "python-chess", "numpy", "tqdm"),
    volumes={"/data": data_vol},
    gpu="A100",
    timeout=108000,
)
def train_on_modal(
    pgn_path: str = "/data/train.pgn",
    epochs: int = 3,
    batch_size: int = 128,
    cache_path: str = "/data/pgn_cache.pt",
    checkpoint_dir: str = "/data/checkpoints",
    num_workers: int = 4,
    precompute: bool = True,
    lr: float = 3e-4,
):
    """
    Entrypoint that runs the existing supervised trainer inside Modal.
    The repository code (src.training.supervised_training.tiny_supervised_train)
    will be imported and executed inside the Modal container.
    """
    print("Modal: importing and starting supervised trainer...")
    # import inside function so Modal's image environment picks up the repo runtime
    from src.training.supervised_training import tiny_supervised_train

    tiny_supervised_train(
        pgn_path=pgn_path,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        num_workers=num_workers,
        precompute=precompute,
        cache_path=cache_path,
        val_split=0.05,
        checkpoint_dir=checkpoint_dir,
        save_every=1,
    )


@stub.local_entrypoint()
def main():
    """
    Local CLI to either launch Modal remote training or run trainer locally.
    Example local run:
      python -m src.train --pgn /path/to/games.pgn --epochs 2 --batch-size 64 --no-modal
    Example Modal run (will mount persisted volume 'chess-data-vol'):
      python -m src.train --pgn /data/train.pgn --use-modal
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", required=True, help="Path to PGN file (local or on /data for Modal)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-path", default="/data/pgn_cache.pt")
    parser.add_argument("--checkpoint-dir", default="/data/checkpoints")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-precompute", dest="precompute", action="store_false")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--use-modal", action="store_true", help="Run training on Modal instead of locally")
    args = parser.parse_args()

    if args.use_modal:
        print("Starting remote Modal training...")
        # pass args through to the Modal function
        train_on_modal.remote(
            pgn_path=args.pgn,
            epochs=args.epochs,
            batch_size=args.batch_size,
            cache_path=args.cache_path,
            checkpoint_dir=args.checkpoint_dir,
            num_workers=args.num_workers,
            precompute=args.precompute,
            lr=args.lr,
        )
    else:
        print("Running training locally (no Modal)...")
        from src.training.supervised_training import tiny_supervised_train

        tiny_supervised_train(
            pgn_path=args.pgn,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            num_workers=args.num_workers,
            precompute=args.precompute,
            cache_path=args.cache_path,
            val_split=0.05,
            checkpoint_dir=args.checkpoint_dir,
            save_every=1,
            )
            


            # # the PGN path must be on the Modal volume (mounted at /data inside the container)
# python -m src.train --pgn /data/train.pgn --use-modal --epochs 3 --batch-size 128 --cache-path /data/pgn_cache.pt --checkpoint-dir /data/checkpoints