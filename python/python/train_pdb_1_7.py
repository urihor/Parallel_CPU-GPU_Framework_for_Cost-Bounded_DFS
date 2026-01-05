import os
import time
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from pdb_radix import build_radix_weights
from pdb_dataset import PDB1to7Dataset
from pdb_models import PDB1to7Net

# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------
# One clean format with timestamp + level, suitable for long runs.
logging.basicConfig(
    level=logging.INFO,  # change to WARNING to make it even quieter
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def main():
    """
    Full-epoch training script for the 1–7 pattern delta network.

    Pipeline:
      1. Load the 1–7 PDB file (PDB(1..7) values for all abstract states).
      2. Build radix weights for the partial permutation encoding (m = 8).
      3. Create a list of all ranks [0..N-1] and pass it to the dataset.
      4. On each epoch, shuffle the indices (so the dataset iterates them
         in a new order), and train with a DataLoader over PDB1to7Dataset.
      5. The dataset returns (seq, delta) where:
           - seq: length-8 sequence of positions for tiles 1..7 + blank
           - delta: PDB(1..7) - Manhattan(1..7), clipped into [0, max_pdb].
      6. Train PDB1to7Net with cross-entropy over delta classes.
      7. At the end:
           - Save the PyTorch state_dict (.pt)
           - Export an ONNX model for later use (e.g. C++ / ORT).
    """

    # 1. Load PDB file for pattern 1–7
    pdb_path = "python/data/pdb_1_7.bin"  # adjust if your PDB is in a different folder
    if not os.path.exists(pdb_path):
        LOGGER.error("PDB file not found at %s", pdb_path)
        return

    pdb_values = np.fromfile(pdb_path, dtype=np.uint8)
    N = pdb_values.shape[0]
    LOGGER.info("Loaded PDB from %s", pdb_path)
    LOGGER.info("N states = %s", f"{N:,}")

    max_pdb = int(pdb_values.max())
    LOGGER.info("Max PDB value = %d", max_pdb)
    LOGGER.info("Training target: delta = PDB(1–7) - Manhattan(tiles 1..7)")

    # 2. Build radix weights for pattern size 7 (tiles) + blank = 8
    k = 7
    m = k + 1
    w = build_radix_weights(m)
    LOGGER.info("Radix weights (m = %d): %s", m, w)

    # 3. Full index array [0..N-1] (stored as uint32).
    #    The dataset will use these ranks and unrank them via C++.
    indices = np.arange(N, dtype=np.uint32)
    LOGGER.info("Indices array created, shape = %s", indices.shape)

    # 4. Dataset:
    #    PDB1to7Dataset uses:
    #      - `pdb_values` to read PDB(1..7)
    #      - `indices` (ranks) for unranking
    #      - `w`/`m` to reconstruct sequences and Manhattan distances.
    dataset = PDB1to7Dataset(pdb_values, indices, w, m)

    # 5. Model, loss, optimizer
    batch_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    embed_dim = 32
    hidden_dim = 128
    num_classes = max_pdb + 1  # delta in [0..max_pdb]

    model = PDB1to7Net(embed_dim, hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 6. Full-epoch training run
    #    Here we run a single pass over the entire PDB once.
    num_epochs = 1

    for epoch in range(num_epochs):
        LOGGER.info("=== Epoch %d/%d ===", epoch + 1, num_epochs)
        epoch_start = time.time()

        # Global shuffle of indices (dataset reads via indices array).
        # We keep DataLoader(shuffle=False) so the sampling order is
        # fully controlled here.
        np.random.shuffle(indices)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        running_loss = 0.0
        total_seen = 0
        num_batches = 0

        # tqdm progress bar for batches in this epoch.
        for X_batch, y_batch in tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
            mininterval=1.0,  # avoid too-frequent refreshes
        ):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            running_loss += loss_val
            total_seen += X_batch.size(0)
            num_batches += 1

        epoch_time = time.time() - epoch_start
        mean_loss = running_loss / max(1, num_batches)
        LOGGER.info(
            "Epoch %d finished: mean_loss=%.4f, total_seen=%d (N=%d), time=%.1fs",
            epoch + 1,
            mean_loss,
            total_seen,
            N,
            epoch_time,
        )

    LOGGER.info("Full-epoch training run finished.")

    # 7. Save PyTorch weights (for further fine-tuning / export).
    os.makedirs("models", exist_ok=True)
    pt_path = os.path.join("models", "nn_pdb_1_7_delta_full.pt")
    torch.save(model.state_dict(), pt_path)
    LOGGER.info("Saved PyTorch weights to %s", pt_path)

    # 8. Export ONNX model (for C++ / ONNX Runtime or other inference engines)
    model.eval()
    dummy_input = torch.zeros(1, 8, dtype=torch.long).to(device)
    onnx_path = os.path.join("models", "nn_pdb_1_7_delta.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["indices"],
        output_names=["delta_logits"],
        opset_version=17,
        dynamic_axes={
            "indices": {0: "batch"},
            "delta_logits": {0: "batch"},
        },
    )
    LOGGER.info("Exported ONNX model to %s", onnx_path)


if __name__ == "__main__":
    main()
