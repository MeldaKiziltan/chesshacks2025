# Helper scripts for Modal training

This folder contains small helper scripts that wrap common `modal run` commands
for the canonical PyTorch trainer in `src/training/train_cli.py`.

Prerequisites
- Install project dependencies (from repo root):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r chessbot/requirements.txt
```

- Install the Modal CLI and authenticate:

```bash
# Install modal (if not already installed)
pip install modal
# Authenticate
modal login
```

Make scripts executable (one-time):

```bash
chmod +x chessbot/scripts/*.sh
```

Available scripts
- `upload_pgn_modal.sh` — upload local PGN (`chessbot/src/training/train.pgn`) into the Modal volume.
- `train_pytorch_modal.sh` — start the canonical PyTorch training on Modal (`src/training/train_cli.py`).
- `download_best_model_modal.sh` — download the `best_model.pt` checkpoint saved by the run.

Usage examples

```bash
# Upload PGN
chessbot/scripts/upload_pgn_modal.sh

# Start remote PyTorch training
chessbot/scripts/train_pytorch_modal.sh

# Download the best checkpoint
chessbot/scripts/download_best_model_modal.sh
```

Notes
- The canonical trainer uses PGN via `PGNDataset` and saves checkpoints under `/data/checkpoints` in the Modal volume.
- If you want automated CI to run modal jobs, you'll need to provide Modal credentials as repository secrets — I can help add a GitHub Actions workflow if you want.

If you'd like any of these scripts to accept arguments (e.g. custom PGN path, checkpoint dir, epochs), I can add argument parsing and pass-through to the `modal run` invocations.
