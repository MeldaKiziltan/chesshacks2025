from .utils import chess_manager, GameContext
from .player import Player
import torch
import os
import requests # <-- ADDED
import sys # <-- ADDED

# ============================================================================
# INITIALIZATION
# ============================================================================

MODEL_FILE = os.path.join(os.path.dirname(__file__), "..", "best_model.pt")

MODEL_URL = "https://huggingface.co/meldakiziltan/chesshacks2025/resolve/main/best_model.pt"

player = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_player():
    """
    Global singleton for the Player.
    Downloads the model on first call if it doesn't exist.
    """
    global player
    if player is None:
        print(f"[main.py] First call: Initializing player...", file=sys.stderr)
        
        # 4. Check if model file exists
        if not os.path.exists(MODEL_FILE):
            print(f"Warning: Model file not found at {MODEL_FILE}.", file=sys.stderr)
            print(f"Downloading model from {MODEL_URL}...", file=sys.stderr)
            try:
                # 5. Download the file
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status() # Raise an error for bad status codes
                
                with open(MODEL_FILE, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print("Model download complete.", file=sys.stderr)
            
            except Exception as e:
                print(f"FATAL: Failed to download model: {e}", file=sys.stderr)
                print("Initializing with untrained model as a fallback.", file=sys.stderr)
                player = Player(device=device) # Init untrained
                return player

        # 6. Load the model (either existing or just-downloaded)
        try:
            player = Player(model_path=MODEL_FILE, device=device)
            print(f"[main.py] Chess player initialized with model: {MODEL_FILE} (device: {device})", file=sys.stderr)
        except Exception as e:
            print(f"FATAL: Failed to load model from {MODEL_FILE}: {e}", file=sys.stderr)
            print("The file might be corrupt. Initializing untrained model.", file=sys.stderr)
            player = Player(device=device)

    return player
# --- END HACKATHON DOWNLOAD FIX ---


# ============================================================================
# GAME LOOP
# ============================================================================

@chess_manager.entrypoint
def select_move(ctx: GameContext):
    """Called every time the bot needs to make a move.
    """
    # Get the player (it will load/download on the first call)
    p = get_player()
    
    # Get the best move from the neural network
    move, move_probabilities = p.select_move(ctx.board, temperature=1.0)
    
    # Log the move probabilities for the game engine
    ctx.logProbabilities(move_probabilities)
    
    return move


@chess_manager.reset
def reset_game_state(ctx: GameContext):
    """Called at the start of each new game.
    """
    # Ensure player is loaded on reset
    get_player()
    pass