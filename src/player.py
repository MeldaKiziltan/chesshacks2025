import chess
import torch
import torch.nn.functional as F
import random
import datetime
from typing import Tuple, Dict

from .neural_network import Anon
from .features import board_to_tensor, move_to_index


# -------------------------------------------------------------------------
# Debug controls
# -------------------------------------------------------------------------

DEBUG = True           # master switch
DEBUG_POLICY = True    # log policy-based decisions
DEBUG_SEARCH = True    # log alpha–beta search details


def log(msg: str):
    if DEBUG:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[PLAYER {ts}] {msg}")


class Player:
    """Chess player powered by a neural network (policy + value).
    
    Supports:
    - Pure policy selection (no search)
    - Shallow alpha–beta search using value head + policy for move ordering
    """

    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = Anon().to(self.device)
        self.model.eval()

        if model_path:
            self.load_model(model_path)
        else:
            log("Initialized Player with a fresh (untrained) model.")

    def load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        log(f"Loading model from '{path}' on device={self.device} ...")
        checkpoint = torch.load(path, map_location=self.device)
        if "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
            log("Loaded model_state from checkpoint (training script format).")
        else:
            self.model.load_state_dict(checkpoint)
            log("Loaded model_state from raw file (state_dict only).")
        self.model.eval()
        log("Model set to eval() mode.")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def select_move(
        self,
        board: chess.Board,
        sample: bool = False,
        temperature: float = 1.0,
        use_search: bool = False,
        search_depth: int = 2,
    ) -> Tuple[chess.Move, Dict[chess.Move, float]]:
        """
        Select a move using the neural network.

        Args:
            board: current board state
            sample: if True, sample from the policy distribution (no search mode)
            temperature: softmax temperature for policy (no search mode)
            use_search: if True, run a small alpha–beta search using NN value head
            search_depth: depth (in plies) for alpha–beta search

        Returns:
            (chosen_move, {legal_move: probability})
                - move probabilities are always from the root policy head
        """
        if use_search:
            if DEBUG_SEARCH:
                log(f"Selecting move WITH search (depth={search_depth})")
            return self._select_with_search(board, search_depth=search_depth)
        else:
            if DEBUG_POLICY:
                log(f"Selecting move using POLICY only (temperature={temperature}, sample={sample})")
            return self._select_with_policy(board, sample=sample, temperature=temperature)

    # -------------------------------------------------------------------------
    # Pure policy selection (no search)
    # -------------------------------------------------------------------------

    def _select_with_policy(
        self,
        board: chess.Board,
        sample: bool,
        temperature: float,
    ) -> Tuple[chess.Move, Dict[chess.Move, float]]:
        """Use only the policy head to pick a move (no search)."""
        x = board_to_tensor(board).unsqueeze(0).to(self.device)  # (1,18,8,8)

        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(x)
        logits = logits.squeeze(0)

        # Temperature scaling
        if temperature != 1.0:
            logits = logits / max(1e-8, temperature)

        full_probs = F.softmax(logits, dim=0)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")

        legal_indices = [move_to_index(m) for m in legal_moves]
        legal_probs = full_probs[legal_indices]  # (num_legal,)

        if legal_probs.sum().item() <= 0:
            log("[WARN] Policy produced near-zero probabilities for all legal moves; using uniform.")
            chosen = random.choice(legal_moves)
            uniform_p = 1.0 / len(legal_moves)
            move_probs = {m: uniform_p for m in legal_moves}
            return chosen, move_probs

        # Normalize over legal moves
        legal_probs = legal_probs / legal_probs.sum()

        # ---------------------------------------------------------------------
        # PROMOTION OVERRIDE
        # ---------------------------------------------------------------------
        promotion_indices = [i for i, m in enumerate(legal_moves) if m.promotion is not None]

        if promotion_indices:
            # Choose the promotion move with highest NN probability
            best_promo_idx = max(promotion_indices, key=lambda i: legal_probs[i].item())
            chosen_move = legal_moves[best_promo_idx]

            move_probs = {
                m: float(p)
                for m, p in zip(legal_moves, legal_probs.tolist())
            }

            if DEBUG_POLICY:
                top = sorted(move_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
                top_str = ", ".join(f"{m}: {p:.3f}" for m, p in top)
                log(f"[PROMO] Forcing promotion: {chosen_move} | top moves: {top_str}")

            return chosen_move, move_probs
        # ---------------------------------------------------------------------

        # No promotions available: fall back to normal policy logic
        if sample:
            idx_in_legal = torch.multinomial(legal_probs, num_samples=1).item()
        else:
            idx_in_legal = torch.argmax(legal_probs).item()

        chosen_move = legal_moves[idx_in_legal]

        move_probs = {
            m: float(p)
            for m, p in zip(legal_moves, legal_probs.tolist())
        }

        if DEBUG_POLICY:
            top = sorted(move_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
            top_str = ", ".join(f"{m}: {p:.3f}" for m, p in top)
            log(f"Policy-only choice: {chosen_move} | top moves: {top_str}")

        return chosen_move, move_probs



    def _policy_ordered_moves(
        self,
        board: chess.Board,
    ) -> Tuple[list[chess.Move], torch.Tensor]:
        """
        Return legal moves ordered by policy probability (descending), plus their probs.
        """
        x = board_to_tensor(board).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(x)
        logits = logits.squeeze(0)

        full_probs = F.softmax(logits, dim=0)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return [], torch.empty(0, device=self.device)

        legal_indices = [move_to_index(m) for m in legal_moves]
        legal_probs = full_probs[legal_indices]  # (num_legal,)

        # Avoid all-zero issues
        if legal_probs.sum().item() <= 0:
            log("[WARN] Policy produced near-zero probabilities in _policy_ordered_moves; using uniform.")
            legal_probs = torch.ones_like(legal_probs) / len(legal_probs)

        # Sort moves descending by prob
        sorted_pairs = sorted(
            zip(legal_moves, legal_probs.tolist()),
            key=lambda mp: mp[1],
            reverse=True,
        )
        sorted_moves = [m for m, p in sorted_pairs]
        sorted_probs = torch.tensor([p for m, p in sorted_pairs], device=self.device)

        if DEBUG_SEARCH:
            top = sorted_pairs[:3]
            top_str = ", ".join(f"{m}: {p:.3f}" for m, p in top)
            log(f"Move ordering (top 3): {top_str}")

        return sorted_moves, sorted_probs

    def _search_negamax(
    self,
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    ) -> float:
        """
        Negamax with alpha–beta pruning using SIDE-TO-MOVE perspective values.
        """
        if depth == 0 or board.is_game_over():
            return self._evaluate_position(board)  # already side-to-move

        best_value = -float("inf")

        ordered_moves, _ = self._policy_ordered_moves(board)
        if not ordered_moves:
            return self._evaluate_position(board)

        for move in ordered_moves:
            board.push(move)
            score = -self._search_negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            if score > best_value:
                best_value = score
            if best_value > alpha:
                alpha = best_value
            if alpha >= beta:
                break  # cutoff

        return best_value


    def _select_with_search(
    self,
    board: chess.Board,
    search_depth: int = 2,
) -> Tuple[chess.Move, Dict[chess.Move, float]]:
        # Root policy for logging
        x = board_to_tensor(board).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            root_logits, _ = self.model(x)
        root_logits = root_logits.squeeze(0)
        root_probs_full = F.softmax(root_logits, dim=0)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")

        legal_indices = [move_to_index(m) for m in legal_moves]
        legal_probs = root_probs_full[legal_indices]
        if legal_probs.sum().item() <= 0:
            legal_probs = torch.ones_like(legal_probs) / len(legal_probs)
        legal_probs = legal_probs / legal_probs.sum()

        move_probs = {
            m: float(p)
            for m, p in zip(legal_moves, legal_probs.tolist())
        }

        # Negamax at root: side-to-move wants to maximize eval
        best_move = None
        best_score = -float("inf")
        alpha, beta = -float("inf"), float("inf")

        for move in legal_moves:
            board.push(move)
            score = -self._search_negamax(board, search_depth - 1, -beta, -alpha)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

        if best_move is None:
            best_move = random.choice(legal_moves)

        return best_move, move_probs
