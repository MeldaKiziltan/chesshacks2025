from .utils import chess_manager, GameContext
from .player import Player
import torch
import os
import datetime

# ============================================================================
# DEBUG SETTINGS
# ============================================================================

DEBUG = True                   # High-level logs (safe)
DEBUG_MOVES = True            # Log selected move & top probability
DEBUG_VERBOSE = False         # Extreme logs: full policy distribution, FEN, etc.


def log(msg: str):
    """Light wrapper so we can disable debug output easily."""
    if DEBUG:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[DEBUG {timestamp}] {msg}")


# ============================================================================
# INITIALIZATION
# ============================================================================

# Path to model
MODEL_FILE = os.path.join(os.path.dirname(__file__), "..", "best_model.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists(MODEL_FILE):
    log(f"Loading model from {MODEL_FILE} on device={device}")
    player = Player(model_path=MODEL_FILE, device=device)
    log("Model loaded successfully.")
else:
    print("[WARNING] No 'best_model.pt' found — using untrained model.")
    player = Player(device=device)


# ============================================================================
# GAME LOOP
# ============================================================================

@chess_manager.entrypoint
def select_move(ctx: GameContext):
    """Called every time the game engine requests a move from the bot."""
    try:
        if DEBUG_VERBOSE:
            log(f"Current FEN: {ctx.board.fen()}")

        move, move_probabilities = player.select_move(
            ctx.board,
            temperature=1.0,
            use_search=True, search_depth=3
        )

        # Log key info
        if DEBUG_MOVES:
            # Get top prob
            top_move = max(move_probabilities, key=move_probabilities.get)
            log(
                f"Selected move: {move} "
                f"(top policy move: {top_move}, prob={move_probabilities[top_move]:.4f})"
            )

        # Required by competition framework
        ctx.logProbabilities(move_probabilities)
        return move

    except Exception as e:
        print("[ERROR] Exception during move selection:", e)
        # Fallback: choose a random legal move (never crash mid-game)
        fallback = next(iter(ctx.board.legal_moves))
        print(f"[ERROR] Falling back to legal move {fallback}")
        return fallback


@chess_manager.reset
def reset_game_state(ctx: GameContext):
    """Called at the start of each new game."""
    log("Resetting game state...")
    # If you add alpha-beta or MCTS, reset caches here:
    # if hasattr(player, "tt"):
    #     player.tt.clear()
    # log("Transposition table cleared.")
    pass
