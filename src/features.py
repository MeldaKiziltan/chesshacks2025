import chess
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch import Tensor
from .consts import BASE_ACTIONS, ID_TO_PROMO, PIECE_TYPES, PROMO_TO_ID


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

_QUEEN_MOVES = [(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(-1,0),(-2,0),(-3,0),(-4,0),(-5,0),(-6,0),(-7,0),(0,-1),(0,-2),(0,-3),(0,-4),(0,-5),(0,-6),(0,-7),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(-1,1),(-2,2),(-3,3),(-4,4),(-5,5),(-6,6),(-7,7),(-1,-1),(-2,-2),(-3,-3),(-4,-4),(-5,-5),(-6,-6),(-7,-7),(1,-1),(2,-2),(3,-3),(4,-4),(5,-5),(6,-6),(7,-7)]
_KNIGHT_MOVES = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
_PAWN_PROMOTIONS = [(1,1),(1,0),(1,-1)]; _PROMOTION_TYPES = [chess.KNIGHT,chess.BISHOP,chess.ROOK]
_MOVE_TO_LABEL_MAP = {}; _LABEL_TO_MOVE_MAP = {}
def _build_move_maps():
    idx = 0
    for from_sq in range(64):
        from_row, from_col = divmod(from_sq, 8)
        for dr, df in _QUEEN_MOVES:
            to_row, to_col = from_row + dr, from_col + df
            if 0 <= to_row < 8 and 0 <= to_col < 8: move = chess.Move(from_sq, to_row * 8 + to_col); _MOVE_TO_LABEL_MAP[move] = idx; _LABEL_TO_MOVE_MAP[idx] = move
            idx += 1
        for dr, df in _KNIGHT_MOVES:
            to_row, to_col = from_row + dr, from_col + df
            if 0 <= to_row < 8 and 0 <= to_col < 8: move = chess.Move(from_sq, to_row * 8 + to_col); _MOVE_TO_LABEL_MAP[move] = idx; _LABEL_TO_MOVE_MAP[idx] = move
            idx += 1
        for dr, df in _PAWN_PROMOTIONS:
            for prom_piece in _PROMOTION_TYPES:
                to_row, to_col = from_row + dr, from_col + df
                if 0 <= to_row < 8 and 0 <= to_col < 8: move = chess.Move(from_sq, to_row * 8 + to_col, promotion=prom_piece); _MOVE_TO_LABEL_MAP[move] = idx; _LABEL_TO_MOVE_MAP[idx] = move
                idx += 1
_build_move_maps()


def move_to_index(move: chess.Move) -> int:
    """
    Map a python-chess Move to a flat action index in [0, ACTION_SIZE).

    Layout:
        idx = (from * 64 + to) + promo_id * BASE_ACTIONS
    where promo_id encodes the promotion piece (or none).
    """
    base = move.from_square * 64 + move.to_square
    promo_id = PROMO_TO_ID.get(move.promotion, 0)
    return base + promo_id * BASE_ACTIONS


def index_to_move(idx: int) -> chess.Move:
    """
    Inverse of move_to_index: map index back to a chess.Move.

    If promo_id == 0, it's a normal move.
    If > 0, it's a promotion move.
    """
    promo_id, base = divmod(idx, BASE_ACTIONS)
    from_sq, to_sq = divmod(base, 64)
    promotion_piece = ID_TO_PROMO.get(promo_id, None)
    return chess.Move(from_sq, to_sq, promotion=promotion_piece)
