import chess
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch import Tensor
from typing import Tuple
from nueral_network import Anon
from consts import ACTION_SIZE, PIECE_TYPES


# -------------------------
# Board encoding
# -------------------------

# We need 18 Channels/features:
#  6 piece types (Pawn, Rook, Knight, Bishop, Queen, King) * 2 colors = 12
#  Whose turn it is (1)
#  castling rights (Queen Side, King Side) * 2 Colors = 4
#  en-passant file plane (1)


def set_pieces_on_planes(board: chess.Board, planes: np.ndarray):
    """
    Planes 0 ... 5 represent white pieces in order: pawn, knight, bishop, rook, queen, king.
    Planes 6 ... 11 represent black pieces in order: pawn, knight, bishop, rook, queen, king.
    """
    for i, piece_type in enumerate(PIECE_TYPES):
        for square in board.pieces(piece_type, chess.WHITE):
            row, file = divmod(square, 8)
            planes[i, row, file] = 1.0

        for square in board.pieces(piece_type, chess.BLACK):
            row, file = divmod(square, 8)
            planes[6 + i, row, file] = 1.0


def set_side_to_move_on_planes(board: chess.Board, planes: np.ndarray):
    """
    Configures plane to show whose turn it is
    If it is white to play, all rows and columns on plane 12 would be set 1 one, else set to 0.
    """
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0
    elif board.turn == chess.BLACK:
        planes[12, :, :] = 0.0


def set_castling_rights_on_planes(board: chess.Board, planes: np.ndarray):
    """
    Configures respective planes to show castling rights.
    Sets all rows and columns to 1 if all squares are available.

    Plane 13 represents King side castling rights for white.
    Plane 14 represents Queen side castling rights for white.
    Plane 15 represents King side castling rights for black.
    Plane 16 represents Queen side castling rights for black.
    """
    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0


def set_enpessant_rights_on_planes(board: chess.Board, planes: np.ndarray):
    """
    Configures plane 17 to show en-passant availability.
    If en-passant is possible, set the file (column) to 1.
    """
    if board.ep_square is not None:
        _, file = divmod(board.ep_square, 8)
        planes[17, :, file] = 1.0


def board_to_tensor(board: chess.Board) -> Tensor:
    """Convert python-chess Board to FloatTensor shape (18, 8, 8).
    ---- 18 planes of an 8 x 8 chessboard ---

    Channels order:
      0..5   : white pawn, knight, bishop, rook, queen, king
      6..11  : black pawn, knight, bishop, rook, queen, king
      12     : side to move (all ones if white to move else zeros)
      13..16 : castling rights: white K, white Q, black K, black Q (all ones on all squares if available)
      17     : en-passant file (which file is target square), else zeros
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    set_pieces_on_planes(board, planes)
    set_side_to_move_on_planes(board, planes)
    set_castling_rights_on_planes(board, planes)
    set_enpessant_rights_on_planes(board, planes)

    return torch.from_numpy(planes)


# -------------------------
# Move Index Helpers
# -------------------------

# The policy head(intuition) outputs a vector of length 4096 (same as ACTION_SIZE). [-4.0, 1.5, -1.7, 6, 0, ...]
# Each entry in this vector corresponds to one possible move and contains a raw score
# representing how likely that move is to be the best choice.
#
# To interpret these scores as probabilities, we need to apply a softmax() function over the vector.
# This normalizes all values so they sum to 1 — making each entry a valid probability.
#
# To identify which move a given index represents, use the `index_to_move()` helper function,
# which translates the vector index into an actual chess move (from_square → to_square).


def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def index_to_move(index: int) -> chess.Move:
    """Convert index back to python-chess Move (no promotion info handled).
    Note: for promotion, this simple mapping won't encode promotion piece.
    For many starters that's OK but be aware it's limited.
    """
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)


def choose_move(model, board: chess.Board, device='cpu', sample=False, temperature=1.0) -> chess.Move:
    """Given a board, return a python-chess Move chosen by the network.
    - apply legal move mask
    - either pick argmax or sample from distribution
    """
    model.eval()
    tensor = board_to_tensor(board).unsqueeze(0).to(device)  # [1,C,8,8]
    with torch.no_grad():
        logits, value = model(tensor)
    logits = logits.squeeze(0)  # [ACTION_SIZE]

    # compute probabilities
    if temperature != 1.0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=0)

    # legal mask
    legal = list(board.legal_moves)
    if len(legal) == 0:
        return None
    legal_indices = [move_to_index(m) for m in legal]
    mask = torch.zeros_like(probs)
    mask[legal_indices] = 1.0
    probs = probs * mask
    s = probs.sum().item()
    if s <= 0:
        # numerical safety: fallback to uniform over legal
        chosen = random.choice(legal)
        return chosen
    probs = probs / probs.sum()

    if sample:
        idx = torch.multinomial(probs, num_samples=1).item()
    else:
        idx = torch.argmax(probs).item()
    return index_to_move(idx)


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
        move = choose_move(
            self.model,
            board,
            device=str(self.device),
            sample=sample,
            temperature=temperature,
        )

        if move is None:
            raise ValueError("No legal moves available")

        # Get probabilities for logging
        tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.model(tensor)
        logits = logits.squeeze(0)

        if temperature != 1.0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=0)

        # Create probability dict for legal moves
        legal_moves = list(board.legal_moves)
        move_probs = {}
        for legal_move in legal_moves:
            idx = move_to_index(legal_move)
            move_probs[legal_move] = probs[idx].item()

        return move, move_probs
