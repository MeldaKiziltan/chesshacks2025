import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import time
import sys
import os
import chess
import numpy as np

# --- 1. Modal Setup ---
stub = modal.Stub(name="chess-training-job")
data_vol = modal.Volume.persisted("chess-data-vol")


class Anon(nn.Module):
    def __init__(self, in_channels=18, filters=64, action_size=4672):  # 4672 outputs
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(filters)
        self.res_blocks = nn.ModuleList()
        for _ in range(3):  # Using 3 blocks from your friend's code
            self.res_blocks.append(nn.Sequential(
                nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(filters), nn.ReLU(),
                nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(filters),
            ))
        self.pol_conv = nn.Conv2d(filters, 32, kernel_size=1)
        self.pol_bn = nn.BatchNorm2d(32)
        self.pol_fc = nn.Linear(32 * 8 * 8, action_size)
        self.val_conv = nn.Conv2d(filters, 16, kernel_size=1)
        self.val_bn = nn.BatchNorm2d(16)
        self.val_fc1 = nn.Linear(16 * 8 * 8, 128)
        self.val_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for blk in self.res_blocks:
            out = blk(x);
            x = F.relu(x + out)
        p = F.relu(self.pol_bn(self.pol_conv(x)))
        p = p.view(p.size(0), -1);
        logits = self.pol_fc(p)
        v = F.relu(self.val_bn(self.val_conv(x)))
        v = v.view(v.size(0), -1);
        v = F.relu(self.val_fc1(v))
        value = torch.tanh(self.val_fc2(v)).squeeze(-1)
        return logits, value


PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def vectorize_board(fen: str) -> np.ndarray:
    board = chess.Board(fen);
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    for i, pt in enumerate(PIECE_TYPES):
        for sq in board.pieces(pt, chess.WHITE): r, f = divmod(sq, 8); planes[i, r, f] = 1.0
        for sq in board.pieces(pt, chess.BLACK): r, f = divmod(sq, 8); planes[6 + i, r, f] = 1.0
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    if board.ep_square is not None: file = chess.square_file(board.ep_square); planes[17, :, file] = 1.0
    return planes


_QUEEN_MOVES = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                (0, 7), (-1, 0), (-2, 0), (-3, 0), (-4, 0), (-5, 0), (-6, 0), (-7, 0), (0, -1), (0, -2), (0, -3),
                (0, -4), (0, -5), (0, -6), (0, -7), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (-1, 1),
                (-2, 2), (-3, 3), (-4, 4), (-5, 5), (-6, 6), (-7, 7), (-1, -1), (-2, -2), (-3, -3), (-4, -4), (-5, -5),
                (-6, -6), (-7, -7), (1, -1), (2, -2), (3, -3), (4, -4), (5, -5), (6, -6), (7, -7)]
_KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
_PAWN_PROMOTIONS = [(1, 1), (1, 0), (1, -1)];
_PROMOTION_TYPES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
_MOVE_TO_LABEL_MAP = {};


def _build_move_maps():
    idx = 0
    for from_sq in range(64):
        from_row, from_col = divmod(from_sq, 8)
        for dr, df in _QUEEN_MOVES:
            to_row, to_col = from_row + dr, from_col + df
            if 0 <= to_row < 8 and 0 <= to_col < 8: _MOVE_TO_LABEL_MAP[chess.Move(from_sq, to_row * 8 + to_col)] = idx
            idx += 1
        for dr, df in _KNIGHT_MOVES:
            to_row, to_col = from_row + dr, from_col + df
            if 0 <= to_row < 8 and 0 <= to_col < 8: _MOVE_TO_LABEL_MAP[chess.Move(from_sq, to_row * 8 + to_col)] = idx
            idx += 1
        for dr, df in _PAWN_PROMOTIONS:
            for prom_piece in _PROMOTION_TYPES:
                to_row, to_col = from_row + dr, from_col + df
                if 0 <= to_row < 8 and 0 <= to_col < 8: _MOVE_TO_LABEL_MAP[
                    chess.Move(from_sq, to_row * 8 + to_col, promotion=prom_piece)] = idx
                idx += 1


_build_move_maps()


def move_to_label(move_uci: str) -> int:
    try:
        move = chess.Move.from_uci(move_uci)
        if move.promotion == chess.QUEEN: move.promotion = None
        return _MOVE_TO_LABEL_MAP.get(move, 0)
    except:
        return 0


class ChessDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, "r") as f:
            for line in f:
                try:
                    fen, uci = line.strip().split("|")
                    vector = vectorize_board(fen)
                    label = move_to_label(uci)
                    # We only yield what we have: position and move
                    yield torch.tensor(vector, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
                except Exception:
                    pass


# --- 5. The Training Step---
def supervised_step(model, optimizer, batch_positions, batch_targets, batch_outcomes=None):

    model.train()
    logits, values = model(batch_positions)
    loss_policy = F.cross_entropy(logits, batch_targets)

    loss_value = torch.tensor(0.0).to(batch_positions.device)  # Ensure it's on the GPU
    if batch_outcomes is not None:
        loss_value = F.mse_loss(values, batch_outcomes)

    loss = loss_policy + loss_value
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), loss_policy.item(), loss_value.item()


# --- 6. The Validation Function ---
def validate_model(model, loader, device):
    print("--- Running Validation ---")
    model.eval()
    correct_predictions, total_predictions = 0, 0
    val_start_time = time.monotonic()
    with torch.no_grad():
        for i, (vectors, labels) in enumerate(loader):
            vectors, labels = vectors.to(device), labels.to(device)
            logits, _ = model(vectors)  # We only care about logits
            _, predicted = torch.max(logits.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            if i % 100 == 0: print('.', end='', file=sys.stderr); sys.stderr.flush()
    val_end_time = time.monotonic()
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\nValidation complete in {(val_end_time - val_start_time):.2f} seconds. Accuracy: {accuracy:.2f}%")
    return accuracy


# --- 7. The Modal Training Function ---
@stub.function(
    image=modal.Image.debian_slim().pip_install("torch", "python-chess", "numpy"),
    volumes={"/data": data_vol},
    gpu="A100",
    timeout=108000  # 30 hours
)
def train_on_modal():
    print("--- Starting Training on Modal ---")

    # === CONFIGURATION ===
    TRAIN_FILE = "/data/train_set.txt"
    VAL_FILE = "/data/validation_set.txt"
    MODEL_SAVE_PATH = "/data/chess_bot_model.pth"

    # You MUST update these numbers from `wc -l`
    TRAIN_SET_SIZE = 104434711  # (UPDATE THIS)
    VAL_SET_SIZE = 2131321  # (UPDATE THIS)

    BATCH_SIZE = 1024
    LEARNING_RATE = 3e-4  #
    NUM_EPOCHS = 3
    # =======================

    device = "cuda"
    train_dataset = ChessDataset(TRAIN_FILE)
    val_dataset = ChessDataset(VAL_FILE)
    # num_workers=4 speeds up data loading
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    model = Anon().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        report_every_n_batches = (TRAIN_SET_SIZE // BATCH_SIZE) // 100
        if report_every_n_batches == 0: report_every_n_batches = 1

        print(f"\n--- Starting Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        epoch_start_time = time.monotonic()

        for i, (vectors, labels) in enumerate(train_loader):
            vectors, labels = vectors.to(device), labels.to(device)

            # We pass `batch_outcomes=None`, so it only trains the policy head.
            loss, p_loss, v_loss = supervised_step(model, optimizer, vectors, labels, batch_outcomes=None)
            total_loss += loss

            if (i + 1) % report_every_n_batches == 0:
                percent_complete = ((i + 1) * BATCH_SIZE / TRAIN_SET_SIZE) * 100
                avg_loss = total_loss / (i + 1)
                print(f"[Epoch {epoch + 1}, {percent_complete:.1f}%] Train Loss: {avg_loss:.4f} (p_loss: {p_loss:.4f})")

        epoch_end_time = time.monotonic()
        print(f"--- Epoch {epoch + 1} Finished ---")
        print(f"Time: {(epoch_end_time - epoch_start_time):.2f} seconds")

        current_accuracy = validate_model(model, val_loader, device)

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            print(f"New best model! Accuracy: {best_accuracy:.2f}%. Saving to {MODEL_SAVE_PATH}...")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            data_vol.commit()  # Commit the save
        elif epoch > 0:
            print(f"Validation accuracy did not improve. Best was {best_accuracy:.2f}%. Stopping early.")
            break
    print("--- Training Finished ---")


# --- 8. Local Entrypoint ---
@stub.local_entrypoint()
def main():
    print("Starting remote training job on Modal...")
    train_on_modal.remote()