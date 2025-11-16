import chess
import torch
import math
from .neural_network import Anon
from .features import board_to_tensor
from typing import Dict, Tuple, Optional

# --- HEURISTICS: MATERIAL VALUES ---
# Your request: focus on the importance of capturing pieces.
# This is our "ground truth" for the evaluation.
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}
# A huge value for checkmate, which the search will find.
MATE_SCORE = 1000000
MATE_THRESHOLD = MATE_SCORE - 100 # Scores above this are mates

class Searcher:
    def __init__(self, model: Anon, device: str):
        """
        Initializes the searcher.
        """
        self.model = model
        self.device = device
        self.nodes_searched = 0
        self.default_depth = 5 # <-- INCREASED to 5-ply. Will find/block Mates-in-2
        
        # Transposition Table (cache)
        # This stores (Zobrist hash, depth) -> (score, best_move)
        self.transposition_table: Dict[int, Tuple[int, Optional[chess.Move]]] = {}

    def get_material_score(self, board: chess.Board) -> int:
        """Calculates the raw material score of the board."""
        score = 0
        for piece_type, value in PIECE_VALUES.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value
        
        # Return score from the perspective of the side to move
        return score if board.turn == chess.WHITE else -score

    def evaluate(self, board: chess.Board) -> int:
        """
        This is the new evaluation heuristic you asked for.
        It combines the NN's "intuition" with a *heavy* focus on material.
        """
        if board.is_checkmate():
            return -MATE_SCORE  # Checkmated, this is a terrible position
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0 # Draw
            
        # 1. Get the NN's "intuition"
        tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _logits, nn_value = self.model(tensor)
        
        # Convert NN value (tanh, -1 to 1) to a centipawn score
        nn_score = int(nn_value.item() * 300) # Reduced multiplier
        
        # 2. Get the "ground truth" material score
        material_score = self.get_material_score(board)
        
        # --- THIS IS THE FIX ---
        # The bot will now care *much* more about material than the NN's opinion.
        # It will no longer "undermine" captures.
        final_score = int(nn_score * 0.2 + material_score * 0.8)
        # --- END FIX ---
        
        return final_score

    def find_best_move(self, board: chess.Board, policy_map: Dict[chess.Move, float], time_limit_ms: int):
        """
        Public entrypoint to start the search.
        """
        self.nodes_searched = 0
        self.transposition_table.clear()
        
        # NOTE: A real time-based search is complex. For now, we use a
        # fixed depth, which is more reliable for a hackathon.
        search_depth = self.default_depth
        
        best_move, best_score = self.negamax(
            board,
            search_depth,
            -MATE_SCORE - 1,
            MATE_SCORE + 1,
            policy_map # Pass the root policy map for move ordering
        )
        
        print(f"[Searcher] Searched {self.nodes_searched} nodes. Best move: {best_move.uci()} Score: {best_score}")
        
        # Failsafe: If search fails (returns None), pick highest policy move
        if best_move is None:
            print("[Searcher] WARNING: Search returned None. Falling back to policy.")
            best_move = max(policy_map, key=policy_map.get)

        return best_move

    def order_moves(self, board: chess.Board, policy_map: Optional[Dict[chess.Move, float]]) -> list[chess.Move]:
        """
        Orders legal moves to make alpha-beta pruning more effective.
        This is the "heuristic" you asked for.
        """
        legal_moves = list(board.legal_moves)
        
        def get_move_score(move: chess.Move) -> int:
            score = 0
            
            # 1. HEURISTIC: Prioritize captures (Quiescence)
            if board.is_capture(move):
                # "Most Valuable Victim - Least Valuable Attacker"
                victim_val = PIECE_VALUES.get(board.piece_type_at(move.to_square), 0)
                attacker_val = PIECE_VALUES.get(board.piece_type_at(move.from_square), 0)
                score += 20000 + (victim_val - attacker_val) # Prioritize good captures
            
            # 2. HEURISTIC: Use the NN's "intuition" *if* we have it
            if policy_map:
                score += int(policy_map.get(move, 0) * 100) # Use NN policy as a hint
            
            # 3. HEURISTIC: Promotions are good
            if move.promotion:
                score += PIECE_VALUES.get(move.promotion, 100)
                
            return score

        return sorted(legal_moves, key=get_move_score, reverse=True)

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, policy_map: Optional[Dict[chess.Move, float]]) -> (chess.Move, int):
        """
        The main Alpha-Beta search function.
        """
        self.nodes_searched += 1
        
        # Check for game over
        if board.is_game_over(claim_draw=True):
            if board.is_checkmate():
                # We got checkmated, this is a losing path
                # Add depth to find the *fastest* mate
                return (None, -MATE_SCORE + (self.default_depth - depth)) 
            return (None, 0) # This path is a draw

        # --- HEURISTIC 1: Reached depth limit, switch to capture search ---
        if depth == 0:
            score = self.quiescence_search(board, alpha, beta)
            return (None, score)
        
        # Check transposition table
        # We combine hash and depth for a unique key
        hash_key = chess.zobrist_hash(board) ^ (depth * 1000) 
        if hash_key in self.transposition_table:
            cached_score, cached_move = self.transposition_table[hash_key]
            # If we have a valid move, we can use it for move ordering
            if cached_move is not None:
                policy_map = {cached_move: 1.0}

        best_move = None
        best_score = -MATE_SCORE - 1

        # --- HEURISTIC 2: Order moves based on policy and captures ---
        ordered_moves = self.order_moves(board, policy_map)
        
        for move in ordered_moves:
            board.push(move)
            # Search the child node. Score is from the *opponent's* perspective.
            # We pass policy_map=None to deeper nodes; ordering will be by captures
            _move, score = self.negamax(board, depth - 1, -beta, -alpha, None)
            score = -score # Flip score back to our perspective
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            
            # Alpha-Beta Pruning
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break # Prune this branch
        
        # Store in transposition table
        self.transposition_table[hash_key] = (best_score, best_move)
        return best_move, best_score

    def quiescence_search(self, board: chess.Board, alpha: int, beta: int) -> int:
        """
        A special, fast search that *only* looks at captures to
        avoid "horizon effect" blunders. This is the heuristic
        that understands the "importance of capturing pieces."
        """
        self.nodes_searched += 1

        # --- HEURISTIC 3: Use the (new) Value head as the baseline ---
        stand_pat_score = self.evaluate(board)
        
        if stand_pat_score >= beta:
            return beta # Fail-high
        
        alpha = max(alpha, stand_pat_score)

        # Generate *only* legal capture moves
        capture_moves = list(board.generate_legal_captures())
        
        # Order them by "Most Valuable Victim - Least Valuable Attacker"
        capture_moves = sorted(capture_moves, key=lambda m: 
            PIECE_VALUES.get(board.piece_type_at(m.to_square), 0) - 
            PIECE_VALUES.get(board.piece_type_at(m.from_square), 0), 
            reverse=True
        )

        for move in capture_moves:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta # Prune
            alpha = max(alpha, score)

        return alpha