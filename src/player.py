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
            logits, value = self.model(x)   # logits: (1,4096), value: (1,) or (1,1)
        logits = logits.squeeze(0)          # (4096,)

        # Temperature scaling
        if temperature != 1.0:
            logits = logits / max(1e-8, temperature)

        full_probs = F.softmax(logits, dim=0)  # (4096,)

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

        legal_probs = legal_probs / legal_probs.sum()

        # Choose move
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
            # Sort to display top few moves
            top = sorted(move_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
            top_str = ", ".join(f"{m}: {p:.3f}" for m, p in top)
            log(f"Policy-only choice: {chosen_move} | top moves: {top_str}")

        return chosen_move, move_probs

    # -------------------------------------------------------------------------
    # Alpha–beta search using NN value head + policy ordering
    # -------------------------------------------------------------------------

    def _evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate a position with the value head.

        IMPORTANT:
        - Value is trained as final game result from WHITE's perspective:
          +1 = white win, -1 = black win, 0 = draw.
        - So here we always return "how good this is for White", regardless of side to move.
        """
        # Terminal positions: use true result directly, matching your PGN labeling
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1.0   # white won
            elif result == "0-1":
                return -1.0  # black won
            else:
                return 0.0   # draw

        # Non-terminal: ask the NN
        x = board_to_tensor(board).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, value = self.model(x)  # ignore policy
        return float(value.view(()).item())  # scalar: "good for White"


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

    def _search_minimax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        """
        Minimax with alpha–beta pruning using WHITE-perspective values.

        - If it's White to move: maximize the eval (good for White).
        - If it's Black to move: minimize the eval (bad for White = good for Black).
        """
        if depth == 0 or board.is_game_over():
            return self._evaluate_position(board)  # WHITE-perspective scalar

        ordered_moves, _ = self._policy_ordered_moves(board)
        if not ordered_moves:
            # No legal moves (checkmate/stalemate) – eval handles it
            return self._evaluate_position(board)

        if board.turn == chess.WHITE:
            # Maximizing player
            best_value = -float("inf")
            for move in ordered_moves:
                board.push(move)
                score = self._search_minimax(board, depth - 1, alpha, beta)
                board.pop()

                if score > best_value:
                    best_value = score
                if best_value > alpha:
                    alpha = best_value
                if alpha >= beta:
                    break  # beta cutoff
            return best_value
        else:
            # Minimizing player (Black)
            best_value = float("inf")
            for move in ordered_moves:
                board.push(move)
                score = self._search_minimax(board, depth - 1, alpha, beta)
                board.pop()

                if score < best_value:
                    best_value = score
                if best_value < beta:
                    beta = best_value
                if alpha >= beta:
                    break  # alpha cutoff
            return best_value


    def _select_with_search(
        self,
        board: chess.Board,
        search_depth: int = 2,
    ) -> Tuple[chess.Move, Dict[chess.Move, float]]:
        """
        Use alpha–beta minimax search guided by NN policy & value.

        - Policy head used for move ordering at each node.
        - Value head used to evaluate leaf positions (WHITE-perspective).
        - Root move is chosen by maximizing or minimizing based on side to move.
        """
        # Root policy for logging (same as policy-only path)
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

        # Now search for best move according to WHITE-perspective eval
        best_move = None
        alpha, beta = -float("inf"), float("inf")

        if board.turn == chess.WHITE:
            # White wants to maximize eval
            best_score = -float("inf")
            for move in legal_moves:
                board.push(move)
                score = self._search_minimax(board, search_depth - 1, alpha, beta)
                board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move
                if score > alpha:
                    alpha = score
        else:
            # Black wants to minimize eval
            best_score = float("inf")
            for move in legal_moves:
                board.push(move)
                score = self._search_minimax(board, search_depth - 1, alpha, beta)
                board.pop()

                if score < best_score:
                    best_score = score
                    best_move = move
                if score < beta:
                    beta = score

        if best_move is None:
            best_move = random.choice(legal_moves)

        return best_move, move_probs

