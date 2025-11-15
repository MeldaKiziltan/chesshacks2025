import os
import datetime

import torch
from huggingface_hub import hf_hub_download

from .utils import chess_manager, GameContext
from .player import Player


# ============================================================================
# DEBUG
# ============================================================================

DEBUG = True

def log(msg: str):
    if DEBUG:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[MAIN {ts}] {msg}")


# ============================================================================
# MODEL CONFIG
# ============================================================================

# For reference, direct file URL (for your own sanity):
#   https://huggingface.co/VallyDev/everest/resolve/main/best_model.pt

# Hugging Face repo + filename
HF_REPO_ID = os.getenv("HF_REPO_ID", "VallyDev/everest")
HF_MODEL_FILE = os.getenv("HF_MODEL_FILE", "best_model.pt")

# Fallback local path (optional)
LOCAL_MODEL_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "best_model.pt")
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# PLAYER INITIALIZATION
# ============================================================================

# Try Hugging Face first
player = None
try:
    log(f"Trying to download model from Hugging Face: repo={HF_REPO_ID}, file={HF_MODEL_FILE}")
    ckpt_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILE,
    )
    log(f"Downloaded model to local path: {ckpt_path}")
    player = Player(model_path=ckpt_path, device=device)
    log(f"Chess player initialized from Hugging Face (device={device})")

except Exception as e:
    log(f"Failed to load model from Hugging Face: {e}")
    # Fallback: try local file
    if os.path.exists(LOCAL_MODEL_FILE):
        log(f"Falling back to local model file: {LOCAL_MODEL_FILE}")
        player = Player(model_path=LOCAL_MODEL_FILE, device=device)
        log(f"Chess player initialized from local file (device={device})")
    else:
        log("No local model found. Initializing untrained model.")
        player = Player(device=device)


# ============================================================================
# GAME LOOP
# ============================================================================

@chess_manager.entrypoint
def select_move(ctx: GameContext):
    """Called every time the bot needs to make a move."""
    move, move_probabilities = player.select_move(ctx.board, temperature=1.0, use_search=True, search_depth=3)
    ctx.logProbabilities(move_probabilities)
    return move


@chess_manager.reset
def reset_game_state(ctx: GameContext):
    """Called at the start of each new game."""
    log("Resetting game state...")
    # If you add search caches / transposition tables to Player, clear them here.
    pass
