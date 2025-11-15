# Save this file as chessbot/train_modal.py
# --- THIS IS THE CORRECTED VERSION v4 ---

import modal
import os
import sys

# --- Modal Setup ---
app = modal.App(name="chesshacks-trainer")
data_vol = modal.Volume.from_name("chess-data-vol", create_if_missing=True)

# Define the *base* environment
training_image = (
    modal.Image.debian_slim()
    .pip_install(
        "tensorflow",
        "pandas",
        "numpy",
        "python-chess",
    )
)

# --- Define Paths ---
DATA_DIR = "/data"
LOCAL_DATA_PATH = "chessData.csv"  # The local file
REMOTE_DATA_PATH = os.path.join(DATA_DIR, "chessData.csv")
REMOTE_MODEL_PATH = os.path.join(DATA_DIR, "my_chess_cnn.keras")
LOCAL_MODEL_PATH = "my_chess_cnn.keras" # The file to download

# --- Helper Functions (Copied from your train.py) ---
# (These are unchanged)

def board_to_tensor(board):
    import numpy as np
    import chess
    
    piece_to_channel = {
        (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
    }
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

def parse_evaluation(eval_str):
    eval_str = str(eval_str).strip()
    if '#' in eval_str:
        if eval_str[0] == '#': return 30000 - int(eval_str[1:]) * 100
        elif eval_str[0] == '-': return -30000 + int(eval_str[2:]) * 100
    try:
        return int(eval_str)
    except ValueError:
        return 0

def build_model():
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential([
        layers.Input(shape=(8, 8, 17)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear', name='evaluation')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Modal Functions ---

@app.function(
    image=training_image.add_local_file(
        LOCAL_DATA_PATH, remote_path="/tmp/chessData.csv"
    ),
    volumes={DATA_DIR: data_vol},
)
def upload_data():
    """Copies the mounted dataset into the persistent volume."""
    import shutil
    print("Copying local chessData.csv to persistent volume...")
    if not os.path.exists("/tmp/chessData.csv"):
        print("Error: Local file not mounted correctly.", file=sys.stderr)
        return
        
    shutil.copy("/tmp/chessData.csv", REMOTE_DATA_PATH)
    data_vol.commit()
    print(f"Upload complete. {REMOTE_DATA_PATH} created in volume.")


@app.function(
    image=training_image,
    volumes={DATA_DIR: data_vol},
    gpu="T4",
    timeout=3600
)
def train_model():
    """This is your main training script, running on a remote GPU."""
    import pandas as pd
    import numpy as np
    import chess
    
    print("--- Starting Remote Model Training ---")
    
    if not os.path.exists(REMOTE_DATA_PATH):
        print(f"Error: Dataset not found at {REMOTE_DATA_PATH}", file=sys.stderr)
        print("Please run 'modal run train_modal.py --upload' first.", file=sys.stderr)
        return

    print(f"Loading dataset from {REMOTE_DATA_PATH}...")
    df = pd.read_csv(
        REMOTE_DATA_PATH,
        names=["FEN", "Evaluation"],
        skiprows=1,
        nrows=500000
    )
    print(f"Loaded {len(df)} positions.")

    X_train = []
    y_train = []

    print("Processing data into tensors...")
    for index, row in df.iterrows():
        fen = row['FEN']
        score = parse_evaluation(row['Evaluation'])
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        tensor = board_to_tensor(board)
        X_train.append(tensor)
        y_train.append(score)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"Data processed. Shape of X_train: {X_train.shape}")

    model = build_model()
    model.summary()
    
    print("\n--- Starting Model.fit ---")
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=256,
        validation_split=0.1
    )
    
    model.save(REMOTE_MODEL_PATH)
    data_vol.commit()
    print(f"\n--- Model saved to {REMOTE_MODEL_PATH} in volume ---")


# --- THIS IS THE FIX ---
@app.function(
    volumes={DATA_DIR: data_vol},
    timeout=300
)
def download_model():
    """Downloads the trained model file from the volume."""
    print(f"Downloading {REMOTE_MODEL_PATH} from volume...")
    
    # Reload the volume to see changes from other functions
    data_vol.reload()
    
    if not os.path.exists(REMOTE_MODEL_PATH):
        print(f"Error: Model file not found at {REMOTE_MODEL_PATH}", file=sys.stderr)
        print("Did you run the training function first?", file=sys.stderr)
        return

    try:
        # data_vol.read_file() returns a generator.
        model_bytes_generator = data_vol.read_file(REMOTE_MODEL_PATH)
        
        # We must write the file chunk by chunk.
        with open(LOCAL_MODEL_PATH, "wb") as f:
            for chunk in model_bytes_generator:
                f.write(chunk)
                
        print(f"Model successfully downloaded to {LOCAL_MODEL_PATH}")
            
    except Exception as e:
        print(f"An error occurred during download: {e}", file=sys.stderr)
# --- END OF FIX ---


# --- Local Entrypoint (Your CLI) ---

@app.local_entrypoint()
def main(upload: bool = False, train: bool = False, download: bool = False):
    if not any([upload, train, download]):
        print("No action specified. Use --upload, --train, or --download.")
        print("Example: modal run train_modal.py --upload")
        return

    if upload:
        if not os.path.exists(LOCAL_DATA_PATH):
            print(f"Error: {LOCAL_DATA_PATH} not found in your directory.", file=sys.stderr)
            print("Please download it from Kaggle first.", file=sys.stderr)
        else:
            print("Uploading dataset to Modal...")
            upload_data.remote()
    
    if train:
        print("Starting remote training job on Modal...")
        train_model.remote()

    if download:
        print("Downloading trained model from Modal...")
        download_model.remote()