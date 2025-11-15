import chess
import argparse
import torch
from src.player import Player

def main():
    """
    Loads the trained model and predicts the best move for a given FEN string.
    """
    parser = argparse.ArgumentParser(description="Run inference with a trained chess model.")
    parser.add_argument(
        "--model",
        type=str,
        default="best_model.pt",
        help="Path to the trained model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--fen",
        type=str,
        default=chess.STARTING_FEN,
        help="The FEN string of the board position to evaluate."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top moves to display."
    )
    args = parser.parse_args()

    # Use CPU for local inference by default
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # 1. Initialize the Player
        print(f"Loading model from: {args.model}")
        player = Player(model_path=args.model, device=device)
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model}")
        print("Please make sure 'best_model.pt' is in the same directory.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This might be due to a mismatch in the model definition (Anon) in 'src/neural_network.py'.")
        return

    # 2. Set up the board
    try:
        board = chess.Board(args.fen)
    except ValueError:
        print(f"Error: Invalid FEN string provided.")
        print(f"FEN: {args.fen}")
        return

    print(f"\nEvaluating position: {board.fen()}")

    # 3. Get the model's prediction
    # We use sample=False to get the single best move (argmax)
    move, move_probs = player.select_move(board, sample=False)

    print(f"\n=> Best Move: {move.uci()}")

    # 4. Show top k moves and their probabilities
    print(f"\nTop {args.top_k} moves:")
    # Sort the move_probs dict by probability (value) in descending order
    sorted_moves = sorted(move_probs.items(), key=lambda item: item[1], reverse=True)
    
    for i, (move_obj, prob) in enumerate(sorted_moves[:args.top_k]):
        print(f"  {i+1}. {move_obj.uci():<6} (Probability: {prob:.4%})")

if __name__ == "__main__":
    main()