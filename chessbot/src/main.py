# Replace the contents of chessbot/src/main.py with this

from .utils import chess_manager, GameContext
from chess import Move
import chess
import numpy as np
import tensorflow as tf
import os

# --- Bot Configuration ---
SEARCH_DEPTH = 1  # 3 is fast, 4 is stronger.
MODEL_FILE_NAME = "my_chess_cnn.keras"

# --- 1. Load the Trained ML Model ---

print("Loading trained ML model...")
# Ensure the model path is correct. This assumes my_chess_cnn.keras
# is in the /chessbot folder (where serve.py is run from)
if not os.path.exists(MODEL_FILE_NAME):
    print(f"FATAL ERROR: Model file '{MODEL_FILE_NAME}' not found.")
    print("Please run train.py first!")
    model = None
else:
    model = tf.keras.models.load_model(MODEL_FILE_NAME)
    print("Model loaded successfully.")


# --- 2. Board Vectorization (Must be identical to train.py) ---

piece_to_channel = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}

def board_to_tensor(board):
    tensor = np.zeros((8, 8, 17), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece_to_channel[(piece.piece_type, piece.color)]
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[rank, file, channel] = 1.0
    if board.turn == chess.WHITE:
        tensor[:, :, 12] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[:, :, 13] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[:, :, 14] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[:, :, 15] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[:, :, 16] = 1.0
    return tensor

# --- 3. ML Evaluation Function ---

def nn_evaluate(board):
    """The ML-powered evaluation function."""
    if model is None:
        raise Exception("Model is not loaded.")

    if board.is_checkmate():
        return -float('inf') if board.turn == chess.WHITE else float('inf')
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    tensor = board_to_tensor(board)
    tensor_batch = np.expand_dims(tensor, axis=0)
    score = model.predict(tensor_batch, verbose=0)[0][0]
    return float(score)

# --- 4. Minimax Search Algorithm (with Move Ordering) ---

def get_ordered_moves(board):
    """Scores and sorts legal moves to improve alpha-beta pruning."""
    move_scores = []
    for move in board.legal_moves:
        score = 0
        if board.is_capture(move):
            score += 10
        if move.promotion is not None:
            score += 20
        move_scores.append((move, score))
    move_scores.sort(key=lambda x: x[1], reverse=True)
    return [move for move, score in move_scores]

def minimax(board, depth, alpha, beta, is_maximizing_player):
    """The main Minimax search function."""
    if depth == 0 or board.is_game_over():
        return nn_evaluate(board)

    ordered_moves = get_ordered_moves(board)

    if is_maximizing_player:
        max_eval = -float('inf')
        for move in ordered_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in ordered_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def find_best_move(ctx: GameContext, depth: int):
    """
    Finds the best move and logs probabilities.
    This replaces the 'find_best_move' from our old bot.py.
    """
    board = ctx.board
    best_move = None
    is_maximizing = board.turn == chess.WHITE
    ordered_moves = get_ordered_moves(board)
    
    # Store scores for logging
    move_scores = {}

    if is_maximizing:
        best_eval = -float('inf')
        for move in ordered_moves:
            board.push(move)
            eval = minimax(board, depth - 1, -float('inf'), float('inf'), False)
            board.pop()
            move_scores[move] = eval
            if eval > best_eval:
                best_eval = eval
                best_move = move
    else:
        best_eval = float('inf')
        for move in ordered_moves:
            board.push(move)
            eval = minimax(board, depth - 1, -float('inf'), float('inf'), True)
            board.pop()
            # We want to log scores from the *current* player's POV.
            # Minimax returns scores from White's POV.
            # So if we are Black, a *low* score is good.
            move_scores[move] = eval
            if eval < best_eval:
                best_eval = eval
                best_move = move
    
    # --- Log probabilities (as required by serve.py) ---
    if move_scores:
        # Convert scores to probabilities using softmax for a clean 0-1 range
        scores = np.array(list(move_scores.values()))
        
        # For Black (minimizing), we invert the scores so high is good
        if not is_maximizing:
            scores = -scores
            
        probs = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))
        move_probs_dict = {move: prob for move, prob in zip(move_scores.keys(), probs)}
        ctx.logProbabilities(move_probs_dict)
    else:
        ctx.logProbabilities({})

    return best_move if best_move is not None else ordered_moves[0]


# --- 5. Implement the ChessHacks API ---

@chess_manager.entrypoint
def get_ai_move(ctx: GameContext):
    """
    This is the main function that serve.py will call.
    """
    if model is None:
        raise Exception("Model is not loaded. Cannot make a move.")

    print(f"Cooking move for board: {ctx.board.fen()}")
    print(f"Time left: {ctx.timeLeft}ms")
    
    # We can add logic here to adjust SEARCH_DEPTH based on ctx.timeLeft
    # For now, we use a fixed depth.
    
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available.")

    move = find_best_move(ctx, SEARCH_DEPTH)
    return move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    This gets called when a new game begins.
    Our bot is stateless (for now), so we don't need to do anything.
    """
    print("----- NEW GAME STARTED -----")
    pass