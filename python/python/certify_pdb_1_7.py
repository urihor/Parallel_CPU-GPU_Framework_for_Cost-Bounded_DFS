import os
import time
import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn

from pdb_radix import build_radix_weights
from pdb15_native import build_batch_1_7_c   # <-- must exist similarly to build_batch_8_15_c
from pdb_models import PDB1to7Net

# Path to the 1–7 pattern database (full PDB).
PDB_PATH = "python/data/pdb_1_7.bin"

# Paths to the trained neural models that predict the delta over the PDB.
MODEL_PATHS: List[str] = [
    "models/nn_pdb_1_7_delta_full.pt",
]

# Parameters for the linear congruential generator (LCG) that is used
# to walk through all PDB entries in a pseudo-random but deterministic order.
A_LCG = 1201201
C_LCG = 2531021

# Quantile level used when converting logits to a class (0 < q <= 1).
# For q=0.1 we take a conservative lower quantile of the predicted distribution.
QUANTILE_Q = 0.1

# Batch size for certification (must divide N exactly).
BATCH_SIZE = 2304

# Network architecture hyperparameters (must match training).
EMBED_DIM = 32
HIDDEN_DIM = 128

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("certify_1_7")


# Map logits -> class index based on a quantile of the predicted distribution.
#
# Given:
#   logits: [B, C] raw outputs of the network
#   q:      quantile in (0,1]
#
# We:
#   1. Apply softmax over classes (dim=1) to obtain probabilities.
#   2. Compute the cumulative sum over classes (CDF).
#   3. Count how many classes have CDF < q and use that as an index.
#   4. Clamp the index into [0, C-1].
def quantile_class_from_logits(logits: torch.Tensor, q: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    cdf = torch.cumsum(probs, dim=1)
    idx = (cdf < q).sum(dim=1)
    return idx.clamp_(0, logits.size(1) - 1)


def main():
    # Sanity checks: ensure model files and PDB file exist.
    for p in MODEL_PATHS:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    if not os.path.exists(PDB_PATH):
        raise FileNotFoundError(PDB_PATH)

    # Load PDB values (1–7 pattern) as uint8 from disk.
    pdb_values = np.fromfile(PDB_PATH, dtype=np.uint8)
    N = int(pdb_values.shape[0])
    max_pdb = int(pdb_values.max())
    num_classes = max_pdb + 1

    # Build ranking weights for partial permutations of length m = 8:
    # tiles 1..7 plus the blank.
    m = 8
    w = build_radix_weights(m)

    # We require that BATCH_SIZE divides N exactly so that the LCG-based
    # traversal can cover all states with an integer number of steps.
    if N % BATCH_SIZE != 0:
        raise RuntimeError(f"BATCH_SIZE={BATCH_SIZE} must divide N={N} exactly")

    # Choose device: CUDA if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("device=%s  q=%.3f  N=%s  batch=%d", device, QUANTILE_Q, f"{N:,}", BATCH_SIZE)

    # Load all models into memory (possibly an ensemble).
    models: List[nn.Module] = []
    for path in MODEL_PATHS:
        model = PDB1to7Net(EMBED_DIM, HIDDEN_DIM, num_classes)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        models.append(model)

    # Each step processes exactly BATCH_SIZE entries.
    steps = N // BATCH_SIZE
    LOG.info("steps=%d (covers all states exactly once if LCG is full-period)", steps)

    # Buffers for inputs (X_np) and ground-truth deltas (delta_true_np).
    X_np = np.empty((BATCH_SIZE, m), dtype=np.int32)
    delta_true_np = np.empty((BATCH_SIZE,), dtype=np.int32)

    # Track the maximum overestimation (delta_pred - delta_true) across all states.
    max_over = -10**9

    # Histogram of positive overestimations (index = amount of overestimation).
    # We start with a small size and grow dynamically if needed.
    pos_over_hist = np.zeros(64, dtype=np.int64)

    # Initial state of the LCG used to generate a permutation of all PDB indices.
    state_lcg = 0
    start_state = state_lcg

    t0 = time.time()
    with torch.no_grad():
        for it in range(steps):
            # Fill X_np (inputs) and delta_true_np (true delta) by sampling
            # BATCH_SIZE distinct states from the PDB using the C-level helper.
            state_lcg = build_batch_1_7_c(
                state_lcg, N, A_LCG, C_LCG, w, pdb_values,
                BATCH_SIZE, X_np, delta_true_np
            )

            # Move integer inputs to the chosen device.
            X = torch.from_numpy(X_np).to(device=device, dtype=torch.long)

            # For each model in the ensemble, compute quantile-based class predictions.
            preds = []
            for model in models:
                logits = model(X)
                preds.append(quantile_class_from_logits(logits, QUANTILE_Q).cpu().numpy())

            # Ensemble combination: take the minimum (most conservative) predicted class
            # across all models for each sample.
            delta_pred = np.stack(preds, axis=0).min(axis=0)

            # Overestimation of the delta: predicted - true.
            over = delta_pred.astype(np.int32) - delta_true_np

            # Track the global maximum overestimation.
            local_max = int(over.max())
            if local_max > max_over:
                max_over = local_max

            # Collect statistics for strictly positive overestimations.
            pos = over[over > 0]
            if pos.size:
                mx = int(pos.max())
                # Grow histogram array if needed.
                if mx >= pos_over_hist.size:
                    new = np.zeros(mx + 1, dtype=np.int64)
                    new[:pos_over_hist.size] = pos_over_hist
                    pos_over_hist = new
                # Update histogram via bincount.
                bc = np.bincount(pos, minlength=pos_over_hist.size)
                pos_over_hist[:bc.size] += bc

            # Periodic progress logging.
            if (it + 1) % 20000 == 0:
                LOG.info("progress=%d/%d  current_max_over=%d  elapsed=%.1fs",
                         it + 1, steps, max_over, time.time() - t0)

    # Final summary: maximum overestimation across all PDB states.
    LOG.info("DONE. max_over_delta=%d", max_over)
    LOG.info("positive_over_counts=%s", pos_over_hist[:max_over + 1].tolist())

    # Check whether the LCG returned to its initial state, which would
    # indicate that we have completed a full cycle over all states.
    LOG.info("lcg_cycle_closed=%s", str(state_lcg == start_state))

    # Save the maximum overestimation to a small .npy file for later use
    # (e.g., when constructing over-correction tables).
    np.save(f"cert_max_over_1_7_q{QUANTILE_Q:.3f}.npy", np.array([max_over], dtype=np.int32))
    LOG.info("saved: cert_max_over_1_7_q%.3f.npy", QUANTILE_Q)


if __name__ == "__main__":
    main()
