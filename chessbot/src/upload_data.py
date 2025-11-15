import modal
import os

# --- Configuration ---
# Make sure these files are in the same folder as this script
DATA_FILES = ["train_set.txt", "validation_set.txt"]
# We also need our "quarantined" test set PGN
# DATA_FILES.append("april_2025.pgn")

VOLUME_NAME = "chess-data-vol"
# ---------------------

stub = modal.Stub(name="chess-data-uploader")
data_vol = modal.Volume.persisted(VOLUME_NAME)


@stub.function(
    mounts=modal.Mount.local_dir(".", remote_path="/local_data"),
    volumes={f"/data": data_vol},
    timeout=7200  # 2 hours, in case upload is slow
)
def copy_files():
    import shutil

    for file_name in DATA_FILES:
        local_path = f"/local_data/{file_name}"
        remote_path = f"/data/{file_name}"

        if not os.path.exists(local_path):
            print(f"File not found: {file_name}. Skipping.")
            continue

        print(f"Uploading {file_name} to Modal Volume...")
        shutil.copy(local_path, remote_path)

    print("--- All files uploaded to Volume! ---")
    data_vol.commit()


@stub.local_entrypoint()
def main():
    copy_files.remote()