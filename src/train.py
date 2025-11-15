import argparse
import modal
import os
import sys

# --- Modal Setup ---
app = modal.App(name="chess-training-modal")
data_vol = modal.Volume.from_name("chess-data-vol", create_if_missing=True)
src_dir = os.path.dirname(__file__)

# --- MODIFIED: Hard-coded path to your combined file ---
# This path is STILL CORRECT, as the data stays in src/training
LOCAL_PGN_PATH = os.path.join(src_dir, "training", "combined_train.pgn")
REMOTE_PGN_PATH = "/data/train.pgn"

# --- Define Model Paths ---
REMOTE_MODEL_PATH = "/data/checkpoints/best_model.pt"
LOCAL_MODEL_PATH = "best_model.pt" # Will save to your current directory


# --- RE-ADDED: Hard-coded upload function ---
@app.function(
    image=modal.Image.debian_slim().add_local_file(
        LOCAL_PGN_PATH, remote_path="/tmp/train.pgn"
    ),
    volumes={"/data": data_vol},
)
def upload_pgn():
    """
    This remote function copies the mounted PGN file from its
    temporary location to the persistent volume.
    """
    import shutil
    
    print(f"Copying PGN from /tmp/train.pgn to {REMOTE_PGN_PATH}...")
    
    if not os.path.exists("/tmp/train.pgn"):
        print(f"Error: Local PGN file at {LOCAL_PGN_PATH} was not mounted correctly.", file=sys.stderr)
        return

    shutil.copy("/tmp/train.pgn", REMOTE_PGN_PATH)
    data_vol.commit()
    print("Upload complete.")
# --- END RE-ADDED FUNCTION ---


# --- Function to download the trained model ---
@app.function(
    volumes={"/data": data_vol},
    image=modal.Image.debian_slim(), # No special libraries needed
)
def download_best_model():
    """
    Reads the 'best_model.pt' file from the persistent volume 
    and returns its content as bytes.
    """
    import os
    import sys
    
    if not os.path.exists(REMOTE_MODEL_PATH):
        print(f"Error: Model not found at {REMOTE_MODEL_PATH}", file=sys.stderr)
        print("Did the training run successfully and save a 'best_model.pt'?", file=sys.stderr)
        return None
    
    print(f"Reading model from {REMOTE_MODEL_PATH}...")
    with open(REMOTE_MODEL_PATH, "rb") as f:
        model_bytes = f.read()
    
    print(f"Read {len(model_bytes)} bytes.")
    return model_bytes
# --- END NEW FUNCTION ---


# --- Training Function (Modified) ---
@app.function(
    image=(
        modal.Image.debian_slim()
        .pip_install("torch", "python-chess", "numpy", "tqdm")
        # --- FIX: Add /root/src to the PYTHONPATH ---
        .env({"PYTHONPATH": "/root:/root/src"})
        .add_local_dir(src_dir, remote_path="/root/src")
    ),
    volumes={"/data": data_vol},
    gpu="A100",
    timeout=86400, # Set to 24h max
)
def train_on_modal(
    pgn_path: str,
    epochs: int,
    batch_size: int,
    cache_path: str,
    checkpoint_dir: str,
    num_workers: int,
    precompute: bool,
    lr: float,
):
    print("Modal: importing and starting supervised trainer...")
    
    # --- Check if PGN file exists ---
    if not os.path.exists(pgn_path):
        print(f"Error: PGN file not found at {pgn_path} in the volume.", file=sys.stderr)
        # --- MODIFIED: Updated error message ---
        print("Please run `modal run src/train.py --upload` first.", file=sys.stderr)
        return
    
    # --- THIS IS THE FIX ---
    # The file is no longer in src.training, it's just in src
    from src.supervised_training import tiny_supervised_train
    # --- END OF FIX ---

    tiny_supervised_train(
        pgn_path=pgn_path,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        num_workers=num_workers,
        precompute=precompute,
        cache_path=cache_path,
        val_split=0.05,
        checkpoint_dir=checkpoint_dir,
        save_every=1,
    )


# --- Local Entrypoint (Modified) ---
@app.local_entrypoint()
def main(
    upload: bool = False,
    # local_path: str = None, # <-- REMOVED
    use_modal: bool = False,
    download: bool = False, 
    pgn: str = REMOTE_PGN_PATH, # Defaults to remote path
    cache_path: str = "/data/pgn_cache.pt",
    checkpoint_dir: str = "/data/checkpoints",
    epochs: int = 3,
    batch_size: int = 128,
    num_workers: int = 4,
    no_precompute: bool = False,
    lr: float = 3e-4,
):
    
    # --- MODIFIED: Upload Logic ---
    if upload:
        if not os.path.exists(LOCAL_PGN_PATH):
            print(f"Error: Local file not found at {LOCAL_PGN_PATH}", file=sys.stderr)
            print("Please make sure your 'combined_train.pgn' file is in the 'src/training' directory.", file=sys.stderr)
            return
        
        print(f"Uploading '{LOCAL_PGN_PATH}' to '{REMOTE_PGN_PATH}'...")
        upload_pgn.remote() # <-- SIMPLIFIED
        return 
    # --- END MODIFIED UPLOAD ---

    # --- Download Logic ---
    if download:
        print(f"Downloading best model from {REMOTE_MODEL_PATH} to {LOCAL_MODEL_PATH}...")
        model_data = download_best_model.remote()
        
        if model_data is None:
            print("Download failed. See remote logs for details.")
            return
            
        with open(LOCAL_MODEL_PATH, "wb") as f:
            f.write(model_data)
        print(f"Successfully saved model to {LOCAL_MODEL_PATH}")
        return
    # --- END Download Logic ---

    precompute = not no_precompute

    if use_modal:
        print("Starting remote Modal training...")
        train_on_modal.remote(
            pgn_path=pgn,
            epochs=epochs,
            batch_size=batch_size,
            cache_path=cache_path,
            checkpoint_dir=checkpoint_dir,
            num_workers=num_workers,
            precompute=precompute,
            lr=lr,
        )
    else:
        print("Running training locally (no Modal)...")
        if not os.path.exists(pgn):
             print(f"Error: Local PGN file not found at {pgn}", file=sys.stderr)
             print("For local runs, please specify the full path, e.g.: --pgn C:\\path\\to\\train.pgn")
             return
        
        # --- THIS IS THE FIX for local training ---
        from src.supervised_training import tiny_supervised_train
        # --- END OF FIX ---

        tiny_supervised_train(
            pgn_path=pgn,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            num_workers=num_workers,
            precompute=precompute,
            cache_path=cache_path,
            val_split=0.05,
            checkpoint_dir=checkpoint_dir,
            save_every=1,
        )