import chess
import torch
import torch.nn.functional as F
import random
from torch import Tensor
from typing import Tuple, Dict
from .neural_network import Anon
from .features import board_to_tensor, move_to_index, index_to_move
from .search import Searcher  # <-- IMPORT THE NEW BRAIN

class Player:
    """
    Chess player that combines a neural network with
    a NegaMax (Alpha-Beta) search algorithm.
    """

    def __init__(self, model_path: str = None, device: str = "cpu"):
        """
        Initialize the Player, model, and searcher.
        """
        self.device = torch.device(device)
        self.model = Anon().to(self.device)
        self.model.eval()

        if model_path:
            self.load_model(model_path)
            
        # --- NEW: Initialize the search brain ---
        self.searcher = Searcher(self.model, self.device)

    def load_model(self, path: str):
        """Load a trained model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
            print(f"Loaded model state from checkpoint: {path}")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"Loaded model state from raw file: {path}")
        self.model.eval() # Ensure model is in eval mode

    def select_move(
        self, board: chess.Board, time_limit_ms: int
    ) -> Tuple[chess.Move, dict]:
        """
        Selects a move by running the NegaMax search.
        
        The NN's policy is used for move ordering.
        The NN's value is used for leaf node evaluation.
        """
        self.model.eval()
        
        # --- 1. Get Policy Map from NN ---
        # We do one forward pass to get the "intuition" (policy)
        # for all legal moves. This is used for move ordering.
        tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _value = self.model(tensor)
        logits = logits.squeeze(0)
        probs = F.softmax(logits, dim=0)

        legal_moves = list(board.legal_moves)
        policy_map: Dict[chess.Move, float] = {}
        
        for move in legal_moves:
            try:
                idx = move_to_index(move) # Use correct 4864 mapping
                policy_map[move] = probs[idx].item()
            except Exception as e:
                # This can happen for castling moves if not mapped
                policy_map[move] = 0.0
                
        # Normalize probabilities
        prob_sum = sum(policy_map.values())
        if prob_sum > 0:
            for move in policy_map:
                policy_map[move] /= prob_sum
        else:
            # Fallback to uniform if all legal moves had 0 prob
            policy_map = {m: 1.0 / len(legal_moves) for m in legal_moves}


        # --- 2. Run the Search ---
        # The searcher will use the policy_map for move ordering
        # and will call self.evaluate() for deep evaluations.
        best_move = self.searcher.find_best_move(board, policy_map, time_limit_ms)

        # --- 3. Apply Promotion Guardrail ---
        # Your hardcoded fix, which is still a great idea!
        if best_move.promotion is not None and best_move.promotion != chess.QUEEN:
            print(f"Warning: Searcher chose underpromotion {best_move.uci()}. Forcing Queen.")
            forced_queen_move = chess.Move(
                best_move.from_square,
                best_move.to_square,
                promotion=chess.QUEEN
            )
            # Check if this forced move is actually legal
            if forced_queen_move in legal_moves:
                best_move = forced_queen_move
            else:
                # This should not happen, but as a fallback, don't change the move
                print(f"Warning: Could not force queen promotion {forced_queen_move.uci()}. Move not legal.")


        return best_move, policy_map