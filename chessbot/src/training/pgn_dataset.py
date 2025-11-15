from pathlib import Path
import chess
import chess.pgn
from torch.utils.data import Dataset
from ..player import move_to_index, board_to_tensor

class PGNDataset(Dataset):
    """A tiny PGN dataset that yields (board_tensor, target_index, outcome)
    outcome is in {-1, 0, +1} from white's perspective (win/draw/loss).
    This code extracts first K moves from first M games to keep it small.
    """

    def __init__(self, pgn_path: str, max_games: int = 50, max_moves_per_game: int = 200):
        self.examples = []  # each: (board_fen, move_index, outcome_float)
        pgn_path = Path(pgn_path)
        if not pgn_path.exists():
            print("PGN not found, dataset will be empty. Pass --demo to run a tiny random demo.")
            return

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

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fen, move_index, outcome = self.examples[idx]
        board = chess.Board(fen)
        return board_to_tensor(board), move_index, outcome