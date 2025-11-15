# Save this file as chessbot/train.py

import pandas as pd
import numpy as np
import chess
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Board Vectorization (17-Channel) ---

# Define the piece-to-channel mapping
piece_to_channel = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

def board_to_tensor(board):
    """
    Converts a chess.Board object to an 8x8x17 tensor.
    Channels 0-11: Piece positions
    Channel 12: Player's turn (1.0 for White)
    Channels 13-16: Castling rights
    """
    tensor = np.zeros((8, 8, 17), dtype=np.float32)
    
    # --- Channels 0-11: Piece positions ---
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece_to_channel[(piece.piece_type, piece.color)]
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[rank, file, channel] = 1.0

    # --- Channel 12: Player's turn ---
    if board.turn == chess.WHITE:
        tensor[:, :, 12] = 1.0

    # --- Channels 13-16: Castling rights ---
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
    """
    Cleans the evaluation string from the dataset.
    """
    eval_str = str(eval_str).strip()
    if '#' in eval_str:
        if eval_str[0] == '#':
            return 30000 - int(eval_str[1:]) * 100
        elif eval_str[0] == '-':
            return -30000 + int(eval_str[2:]) * 100
    try:
        return int(eval_str)
    except ValueError:
        return 0

# --- CNN Model Definition ---

def build_model():
    """
    Defines the CNN architecture for (8, 8, 17) input.
    """
    model = keras.Sequential([
        layers.Input(shape=(8, 8, 17)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear', name='evaluation')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    return model

# --- Main Training Script ---

if __name__ == "__main__":
    # --- !! IMPORTANT !! ---
    # Make sure your CSV file is in the 'chessbot' folder
    # and is named 'chessData.csv'
    # -----------------------
    DATASET_FILE = "chessData.csv" 
    MODEL_SAVE_NAME = "my_chess_cnn.keras"

    print(f"Loading dataset: {DATASET_FILE}...")
    
    df = pd.read_csv(
        DATASET_FILE,
        names=["FEN", "Evaluation"],
        skiprows=1,
        nrows=100000  # Train on 100k samples
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
    
    print("\n--- Starting Model Training ---")
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1
    )
    
    # Save the model in the *root* of the chessbot folder
    model.save(MODEL_SAVE_NAME)
    print(f"\n--- Model saved as {MODEL_SAVE_NAME} ---")
    print("You can now run the bot (via the devtools 'npm run dev')")