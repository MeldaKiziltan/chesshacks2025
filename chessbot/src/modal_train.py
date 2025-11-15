import modal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import time
import sys
import os
import chess

# --- 1. Modal Setup ---
# This tells Modal what Python files our project needs.
# It will package them up with our training job.
stub = modal.Stub(
    name="chess-training-job",
    mounts=modal.Mount.from_local_python_packages(
        "consts",
        "features",
        "neural_network",
        "training.supervised_training"  # For the supervised_step
    )
)
data_vol = modal.Volume.persisted("chess-data-vol")
image = modal.Image.debian_slim().pip_install("torch", "python-chess", "numpy")

# --- 2. The Data & Model Imports ---
# Now we can just import our code, since it's mounted
from neural_network import Anon
from features import board_to_tensor, move_to_index
from training.supervised_training import supervised_step


# --- 3. The Streaming Dataset ---
# This class streams your 106M position file from the Modal Volume
class ChessDataset(IterableDataset):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, "r") as f:
            for line in f:
                try:
                    fen, uci = line.strip().split("|")

                    # Use the 18-plane vectorizer from features.py
                    vector = board_to_tensor(chess.Board(fen))

                    # Use the 4672-label encoder from features.py
                    label = move_to_index(chess.Move.from_uci(uci))

                    # Our "fake" outcome to match the supervised_step function
                    outcome = 0.0

                    yield vector, label, outcome
                except Exception:
                    pass  # Skip any corrupted lines


# --- 4. The Validation Function ---
def validate_model(model, loader, device):
    print("--- Running Validation ---")
    model.eval()
    correct_predictions, total_predictions = 0, 0
    val_start_time = time.monotonic()

    with torch.no_grad():
        for i, (vectors, labels, _) in enumerate(loader):  # Ignore outcome
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


# --- 5. The Modal Training Function ---
@stub.function(
    image=image,
    volumes={"/data": data_vol},
    gpu="A100",
    timeout=108000  # 30 hours
)
def train():
    print("--- Starting Training on Modal ---")

    # === CONFIGURATION ===
    # These paths point to the files you uploaded to the Modal Volume
    TRAIN_FILE = "/data/train_set.txt"
    VAL_FILE = "/data/validation_set.txt"
    MODEL_SAVE_PATH = "/data/best_model.pt"  # This is the file your bot.py will load

    # You MUST update these numbers from `wc -l`
    TRAIN_SET_SIZE = 104434711  # (UPDATE THIS)
    VAL_SET_SIZE = 2131321  # (UPDATE THIS)

    BATCH_SIZE = 1024
    LEARNING_RATE = 3e-4  # Using your preferred LR
    NUM_EPOCHS = 3  # A 30-hour goal
    # =======================

    device = "cuda"
    train_dataset = ChessDataset(TRAIN_FILE)
    val_dataset = ChessDataset(VAL_FILE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    model = Anon().to(device)  # Your `Anon` model
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        report_every_n_batches = (TRAIN_SET_SIZE // BATCH_SIZE) // 100
        if report_every_n_batches == 0: report_every_n_batches = 1

        print(f"\n--- Starting Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        epoch_start_time = time.monotonic()

        for i, (vectors, labels, outcomes) in enumerate(train_loader):
            vectors, labels = vectors.to(device), labels.to(device)
            outcomes = outcomes.to(device)  # Send our "fake" 0.0s

            # --- We call YOUR supervised_step ---
            # It will correctly set loss_value = 0.0
            loss, p_loss, v_loss = supervised_step(model, optimizer, vectors, labels, outcomes)
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
            # We save in the format your `bot_mcts.py` expects
            torch.save({'model_state': model.state_dict()}, MODEL_SAVE_PATH)
            data_vol.commit()  # Save the new best model to the cloud
        elif epoch > 0:
            print(f"Validation accuracy did not improve. Best was {best_accuracy:.2f}%. Stopping early.")
            break
    print("--- Training Finished ---")


# --- 6. Local Entrypoint ---
@stub.local_entrypoint()
def main():
    print("Starting remote training job on Modal...")
    train.remote()