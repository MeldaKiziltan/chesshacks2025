from .utils import chess_manager, GameContext
from .player import Player
import torch
import os

# ============================================================================
# INITIALIZATION
# ============================================================================

# --- NEW: Point to the model you trained ---
# Build a path from this file (src/main.py) up one level to the root
# where 'best_model.pt' is located.
MODEL_FILE = os.path.join(os.path.dirname(__file__), "..", "best_model.pt")
# --- END NEW ---

# Initialize the neural network-powered player at module load time
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- MODIFIED: Pass the model_path to the Player ---
if os.path.exists(MODEL_FILE):
    player = Player(model_path=MODEL_FILE, device=device)
    print(f"Chess player initialized with model: {MODEL_FILE} (device: {device})")
else:
    player = Player(device=device)
    print(f"Warning: 'best_model.pt' not found. Player initialized with new, untrained model.")
# --- END MODIFIED ---


# ============================================================================
# GAME LOOP
# ============================================================================

@chess_manager.entrypoint
def select_move(ctx: GameContext):
    """Called every time the bot needs to make a move.
    
    The neural network evaluates the position and selects a move based on:
    - Policy head: learns which moves are good
    - Value head: evaluates the board position (can be used for search later)
    
    Args:
        ctx: GameContext with current board state and utilities
        
    Returns:
        A legal chess.Move object
    """
    # Get the best move from the neural network
    move, move_probabilities = player.select_move(ctx.board, temperature=1.0)
    
    # Log the move probabilities for the game engine
    ctx.logProbabilities(move_probabilities)
    
    return move


@chess_manager.reset
def reset_game_state(ctx: GameContext):
    """Called at the start of each new game.
    
    Can be used to clear caches or reset model state if needed.
    """
    # Currently a no-op, but here for future extensions
    # (e.g., clearing search trees if you add alpha-beta search)
    pass

