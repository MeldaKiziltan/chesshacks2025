import chess
import numpy as np
import torch
from torch import Tensor
from consts import PIECE_TYPES

def set_pieces_on_planes(board: chess.Board, planes: np.ndarray):
    for i, piece_type in enumerate(PIECE_TYPES):
        for square in board.pieces(piece_type, chess.WHITE):
            row, file = divmod(square, 8); planes[i, row, file] = 1.0
        for square in board.pieces(piece_type, chess.BLACK):
            row, file = divmod(square, 8); planes[6 + i, row, file] = 1.0
def set_side_to_move_on_planes(board: chess.Board, planes: np.ndarray):
    if board.turn == chess.WHITE: planes[12, :, :] = 1.0
    else: planes[12, :, :] = 0.0
def set_castling_rights_on_planes(board: chess.Board, planes: np.ndarray):
    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
def set_enpessant_rights_on_planes(board: chess.Board, planes: np.ndarray):
    if board.ep_square is not None:
        _, file = divmod(board.ep_square, 8); planes[17, :, file] = 1.0
def board_to_tensor(board: chess.Board) -> Tensor:
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    set_pieces_on_planes(board, planes); set_side_to_move_on_planes(board, planes)
    set_castling_rights_on_planes(board, planes); set_enpessant_rights_on_planes(board, planes)
    return torch.from_numpy(planes)

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
_build_move_maps() # Build maps on import

def move_to_index(move: chess.Move) -> int:
    """Converts a chess.Move object to its 4672-index label."""
    if move.promotion == chess.QUEEN: move.promotion = None
    return _MOVE_TO_LABEL_MAP.get(move, 0)

def index_to_move(index: int) -> chess.Move:
    """Converts an integer label back to a chess.Move."""
    return _LABEL_TO_MOVE_MAP.get(index, chess.Move.null())