import chess

BASE_ACTIONS = 64 * 64
PROMO_KINDS = 5  # none, Q, R, B, N
ACTION_SIZE = BASE_ACTIONS * PROMO_KINDS  # 20480
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

PROMO_TO_ID = {
    None: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}

ID_TO_PROMO = {
    0: None,
    1: chess.QUEEN,
    2: chess.ROOK,
    3: chess.BISHOP,
    4: chess.KNIGHT,
}