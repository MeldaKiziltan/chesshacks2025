from pathlib import Path
import chess
import chess.pgn
from torch.utils.data import Dataset, IterableDataset
import torch
from .features import move_to_index, board_to_tensor
from typing import Optional
import os

class PGNDataset(Dataset):
    """A tiny PGN dataset that yields (board_tensor, target_index, outcome)
    outcome is in {-1, 0, +1} from white's perspective (win/draw/loss).
    This code extracts first K moves from first M games to keep it small.
    """

    def __init__(self, pgn_path: str, max_games: int = 1000000, max_moves_per_game: int = 200,
                 precompute: bool = True, cache_path: Optional[str] = None):
        """Load PGN and optionally precompute input tensors.

        Args:
            pgn_path: path to PGN file
            max_games: number of games to read
            max_moves_per_game: limit moves per game
            precompute: if True, convert boards to tensors and store in memory
            cache_path: optional path to save/load precomputed tensors (torch file)
        """
        self.examples = []  # each: (board_fen, move_index, outcome_float)
        self.positions = None  # optional precomputed tensor (N,18,8,8)
        self.targets = None
        self.outcomes = None

        pgn_path = Path(pgn_path)
        if not pgn_path.exists():
            print("PGN not found, dataset will be empty. Pass --demo to run a tiny random demo.")
            return

        # If cache exists, load it for fast startup
        if cache_path and os.path.exists(cache_path):
            data = torch.load(cache_path)
            self.positions = data['positions']
            self.targets = data['targets']
            self.outcomes = data['outcomes']
            return

        # Read PGN and collect examples (store FENs for memory efficiency)
        with pgn_path.open('r', encoding='utf-8', errors='ignore') as fh:
            for g_i in range(max_games):
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                result = game.headers.get('Result', '*')
                outcome = 0.0
                if result == '1-0':
                    outcome = 1.0
                elif result == '0-1':
                    outcome = -1.0
                else:
                    outcome = 0.0

                board = game.board()
                move_count = 0
                for move in game.mainline_moves():
                    if move_count >= max_moves_per_game:
                        break
                    idx = move_to_index(move)
                    # store FEN rather than board obj to reduce memory
                    self.examples.append((board.fen(), idx, outcome))
                    board.push(move)
                    move_count += 1

        # Precompute tensors if requested
        if precompute and len(self.examples) > 0:
            positions = []
            targets = []
            outcomes = []
            for fen, move_idx, outcome in self.examples:
                board = chess.Board(fen)
                t = board_to_tensor(board)
                positions.append(t)
                targets.append(move_idx)
                outcomes.append(outcome)

            # Stack and free examples list to save memory
            self.positions = torch.stack(positions, dim=0)
            self.targets = torch.tensor(targets, dtype=torch.long)
            self.outcomes = torch.tensor(outcomes, dtype=torch.float32)
            self.examples = []

            if cache_path:
                torch.save({'positions': self.positions, 'targets': self.targets, 'outcomes': self.outcomes}, cache_path)

    def __len__(self):
        if self.positions is not None:
            return int(self.positions.size(0))
        return len(self.examples)

    def __getitem__(self, idx):
        if self.positions is not None:
            # return precomputed tensors and labels
            return self.positions[idx], int(self.targets[idx].item()), float(self.outcomes[idx].item())
        fen, move_index, outcome = self.examples[idx]
        board = chess.Board(fen)
        return board_to_tensor(board), move_index, outcome