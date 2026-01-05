import logging
import time
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from pdb_radix import build_radix_weights
from pdb_dataset import PDB1to7Dataset
from pdb_models import PDB1to7Net

# ------------ config ---------------
# Paths and hyperparameters for evaluating a trained 1–7 PDB delta model.

# Path to the 1–7 pattern database (PDB values for tiles 1..7 + blank).
# If in your setup it's "data/..." instead of "python/data/...", update accordingly.
PDB_PATH = "python/data/pdb_1_7.bin"

# Path to the trained PyTorch model (state_dict) that predicts delta = PDB - Manhattan.
MODEL_PATH = "models/nn_pdb_1_7_delta_full.pt"

# Number of random PDB entries to evaluate.
# We sample EVAL_SAMPLES distinct states out of the full PDB.
EVAL_SAMPLES = 200_000

# Batch size for evaluation.
BATCH_SIZE = 4096

# Network architecture hyperparameters (must match the training setup).
EMBED_DIM = 32
HIDDEN_DIM = 128

# ------------ logging --------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# ------------ helpers --------------


def manhattan_1_7_from_seq_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Manhattan distance for tiles 1..7 for a batch of abstract sequences.

    Args:
        x: LongTensor of shape [B, 8].
           For each batch element b:
               x[b, 0..6] = positions (0..15) of tiles 1..7
               x[b, 7]    = position of the blank (ignored here)

    Returns:
        Tensor of shape [B] (int64), where each entry is:
            sum_{t in {1..7}} Manhattan distance of tile t to its goal cell.

    Notes:
        Goal layout is [1..15, 0], so tile t has goal index (t-1).
        We convert to NumPy for simplicity and then back to a tensor on the
        same device as the input x.
    """
    x_np = x.detach().cpu().numpy()
    B = x_np.shape[0]
    res = np.zeros(B, dtype=np.int64)

    for b in range(B):
        total = 0
        for t in range(1, 8):
            pos = int(x_np[b, t - 1])
            r, c = divmod(pos, 4)
            goal_pos = t - 1
            gr, gc = divmod(goal_pos, 4)
            total += abs(r - gr) + abs(c - gc)
        res[b] = total

    return torch.from_numpy(res).to(x.device)


# ------------ main eval ------------


def main():
    """
    Evaluate a trained 1–7 delta model on random samples from the PDB.

    Steps:
      1. Load the 1–7 PDB from disk and gather basic info (N, max value).
      2. Build radix weights (for ranking/unranking) and sample EVAL_SAMPLES
         random indices from the PDB.
      3. Construct a PDB1to7Dataset that, for each sampled index, returns:
            (X, y) where
              X: [8]   positions of tiles 1..7 + blank
              y: scalar true delta = PDB - Manhattan(1..7).
      4. Load the trained PDB1to7Net, move it to device, and run an eval loop:
            - compute logits and CE loss
            - get predicted delta (argmax)
            - collect:
                * cross-entropy loss (per sample)
                * delta accuracy (pred == y)
                * mean absolute error on delta
                * mean absolute error on heuristic h = Manhattan + delta
                * overestimation rate: fraction of samples with h_pred > PDB
      5. Print summary statistics.
    """
    # Check that required files exist.
    if not os.path.exists(PDB_PATH):
        logger.error("PDB file not found at %s", PDB_PATH)
        return
    if not os.path.exists(MODEL_PATH):
        logger.error("Model file not found at %s", MODEL_PATH)
        return

    # 1. Load PDB values into memory (uint8).
    pdb_values = np.fromfile(PDB_PATH, dtype=np.uint8)
    N = pdb_values.shape[0]
    max_pdb = int(pdb_values.max())

    logger.info("Loaded PDB from %s", PDB_PATH)
    logger.info("N states = %s", f"{N:,}")
    logger.info("Max PDB value = %d", max_pdb)

    # 2. Radix weights and random indices (for partial permutation ranking).
    #    We use m = k + 1 = 8 symbols: tiles 1..7 plus the blank.
    k = 7
    m = k + 1
    w = build_radix_weights(m)

    # Randomly sample eval_n distinct PDB indices without replacement.
    rng = np.random.default_rng(seed=12345)
    eval_n = min(EVAL_SAMPLES, N)
    sample_indices = rng.choice(N, size=eval_n, replace=False).astype(np.uint32)

    logger.info("Evaluating on %s random states", f"{eval_n:,}")

    # Dataset wraps (pdb_values, sample_indices, w, m) and returns (X, y) pairs.
    dataset = PDB1to7Dataset(pdb_values, sample_indices, w, m)

    # Use GPU if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 3. Model setup: architecture must match training (EMBED_DIM/HIDDEN_DIM/num_classes).
    num_classes = max_pdb + 1
    model = PDB1to7Net(EMBED_DIM, HIDDEN_DIM, num_classes).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Cross-entropy over all samples (we sum and divide by total at the end).
    criterion = nn.CrossEntropyLoss(reduction="sum")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Eval loop: accumulate statistics over the whole dataset.
    total = 0
    correct_delta = 0
    sum_loss = 0.0
    sum_abs_delta = 0.0
    sum_abs_h = 0.0
    over_count = 0

    t0 = time.time()

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)  # [B, 8]
            y_batch = y_batch.to(device)  # [B]

            # Forward pass: logits over delta classes.
            logits = model(X_batch)       # [B, num_classes]
            loss = criterion(logits, y_batch)

            preds = torch.argmax(logits, dim=1)  # [B], predicted delta class

            # --- stats on delta (PDB - Manhattan) ---
            batch_size = X_batch.size(0)
            total += batch_size

            sum_loss += loss.item()
            correct_delta += (preds == y_batch).sum().item()
            sum_abs_delta += (preds - y_batch).abs().sum().item()

            # --- stats on heuristic h = Manhattan(1..7) + delta ---
            manh = manhattan_1_7_from_seq_batch(X_batch)  # [B]
            h_true = manh + y_batch         # PDB (since y = PDB - manh)
            h_pred = manh + preds           # predicted heuristic

            sum_abs_h += (h_pred - h_true).abs().sum().item()
            over_count += (h_pred > h_true).sum().item()

    elapsed = time.time() - t0

    if total == 0:
        logger.error("No samples evaluated.")
        return

    # Aggregate metrics over all samples.
    mean_loss = sum_loss / total
    delta_acc = correct_delta / total
    mae_delta = sum_abs_delta / total
    mae_h = sum_abs_h / total
    over_rate = over_count / total

    logger.info("=== Eval results on %s samples ===", f"{total:,}")
    logger.info("Mean CE loss (delta)          = %.4f", mean_loss)
    logger.info("Delta accuracy                = %.4f", delta_acc)
    logger.info("Delta MAE                     = %.4f", mae_delta)
    logger.info("Heuristic h MAE               = %.4f", mae_h)
    logger.info("Overestimation rate (h > PDB) = %.6f", over_rate)
    logger.info("Eval time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
