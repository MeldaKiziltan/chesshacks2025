import chess
import torch
import torch.nn.functional as F
import random
from torch import Tensor
from typing import Tuple
from chessbot.src.neural_network import Anon
from .features import board_to_tensor, move_to_index


class Player:
    """Chess player powered by a neural network."""

    def __init__(self, model_path: str = None, device: str = "cpu"):
        """Initialize the Player with a neural network model.

        Args:
            model_path: Path to a saved model checkpoint. If None, initializes a new model.
            device: Device to run the model on ("cpu" or "cuda").
        """
        self.device = torch.device(device)
        self.model = Anon().to(self.device)
        self.model.eval()

        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str):
        """Load a trained model from disk."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded model from {path}")

    def select_move(
        self, board: chess.Board, sample: bool = False, temperature: float = 1.0
    ) -> Tuple[chess.Move, dict]:
        """Select a move using the neural network.

        Args:
            board: Current board state
            sample: If True, sample from the policy distribution. If False, use argmax.
            temperature: Controls randomness (higher = more random)

        Returns:
            Tuple of (selected_move, move_probabilities dict)
        """
        # Single forward pass to get logits and value
        tensor = board_to_tensor(board).unsqueeze(0).to(self.device)  # [1,18,8,8]
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(tensor)
        logits = logits.squeeze(0)

        # apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        probs = F.softmax(logits, dim=0)

        # mask illegal moves
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 0:
            raise ValueError("No legal moves available")

        mask = torch.zeros_like(probs)
        legal_indices = [move_to_index(m) for m in legal_moves]
        mask[legal_indices] = 1.0
        probs = probs * mask
        s = probs.sum().item()
        if s <= 0:
            # Fallback to uniform over legal moves
            chosen = random.choice(legal_moves)
            move_probs = {m: 1.0 / len(legal_moves) for m in legal_moves}
            return chosen, move_probs
        probs = probs / probs.sum()

        # select move
        if sample:
            idx = torch.multinomial(probs, num_samples=1).item()
        else:
            idx = torch.argmax(probs).item()

        # build move probabilities dict
        move_probs = {m: probs[move_to_index(m)].item() for m in legal_moves}

        # translate chosen index to move
        from .features import index_to_move
        chosen_move = index_to_move(idx)
        return chosen_move, move_probs
