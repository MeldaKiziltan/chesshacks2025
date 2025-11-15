from .utils import chess_manager, GameContext
from .player import Player
import torch

# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize the neural network-powered player once at startup
player = None

def initialize_player():
    """Load the model once when the service starts."""
    global player
    device = "cuda" if torch.cuda.is_available() else "cpu"
    player = Player(device=device)
    print(f"✓ Chess player initialized (device: {device})")


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
    global player
    
    if player is None:
        raise RuntimeError("Player not initialized. Call initialize_player() first.")
    
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

