import os
import time
import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn

from pdb_radix import build_radix_weights
from pdb15_native import build_batch_8_15_c
from pdb_models import PDB8to15Net

# Path to the 8–15 pattern database file (full PDB for tiles 8..15 + blank).
PDB_PATH = "python/data/pdb_8_15.bin"

# Paths to the trained neural models (ensemble) that predict the delta over the PDB.
MODEL_PATHS: List[str] = [
    "models/nn_pdb_8_15_delta_lcg_ens0.pt",
    "models/nn_pdb_8_15_delta_lcg_ens1.pt",
    "models/nn_pdb_8_15_delta_lcg_ens2.pt",
    "models/nn_pdb_8_15_delta_lcg_ens3.pt",
]

# LCG parameters used to generate a permutation of all PDB indices.
# IMPORTANT: must match the values that were used during training / data generation.
A_LCG = 1201201
C_LCG = 2531021

# Quantile level for mapping logits to predicted classes (0 < q <= 1).
# For q=0.1 we take a conservative lower quantile of the distribution.
QUANTILE_Q = 0.1

# Batch size for the certification loop. Must divide N exactly.
BATCH_SIZE = 2048  # chosen so that it divides N exactly
EMBED_DIM = 32
HIDDEN_DIM = 128

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("certify_8_15")


# Map logits -> class index using a quantile of the predictive distribution.
#
# Given:
#   logits: [B, C] tensor of raw network outputs
#   q:      quantile in (0,1]
#
# Steps:
#   1. softmax over classes (dim=1) -> probabilities
#   2. cumulative sum (CDF) over classes
#   3. count how many classes have CDF < q  -> index
#   4. clamp index into valid range [0, C - 1]
def quantile_class_from_logits(logits: torch.Tensor, q: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    cdf = torch.cumsum(probs, dim=1)
    idx = (cdf < q).sum(dim=1)
    return idx.clamp_(0, logits.size(1) - 1)


def main():
    # Sanity checks: ensure all model files and the PDB file exist.
    for p in MODEL_PATHS:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    if not os.path.exists(PDB_PATH):
        raise FileNotFoundError(PDB_PATH)

    # Load 8–15 PDB values from disk as uint8.
    pdb_values = np.fromfile(PDB_PATH, dtype=np.uint8)
    N = int(pdb_values.shape[0])
    max_pdb = int(pdb_values.max())
    num_classes = max_pdb + 1

    # We work with m = 9 positions: tiles 8..15 plus the blank.
    # build_radix_weights(m) gives us ranking weights for partial permutations of length m.
    m = 9
    w = build_radix_weights(m)

    # Select device: CUDA if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("device=%s  q=%.3f  N=%s  batch=%d", device, QUANTILE_Q, f"{N:,}", BATCH_SIZE)

    # Load the ensemble of models into memory.
    models: List[nn.Module] = []
    for path in MODEL_PATHS:
        model = PDB8to15Net(EMBED_DIM, HIDDEN_DIM, num_classes)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        models.append(model)

    # Total number of batches such that steps * BATCH_SIZE == N.
    steps = N // BATCH_SIZE
    LOG.info("steps=%d (covers all states exactly once if LCG is full-period)", steps)

    # Buffers for batched inputs (X_np) and ground-truth deltas (delta_true_np).
    X_np = np.empty((BATCH_SIZE, m), dtype=np.int32)
    delta_true_np = np.empty((BATCH_SIZE,), dtype=np.int32)

    # Track the global maximum overestimation (delta_pred - delta_true).
    max_over = -10**9

    # Histogram for positive overestimations (index = amount of overestimation).
    # Starts with a modest size and can grow dynamically.
    pos_over_hist = np.zeros(64, dtype=np.int64)  # enough for small over; will grow if needed

    # Initial state for the LCG. We also keep the starting state to check
    # whether a full cycle is completed.
    state_lcg = 0
    start_state = state_lcg

    t0 = time.time()
    with torch.no_grad():
        for it in range(steps):
            # build_batch_8_15_c:
            #   - updates state_lcg
            #   - fills X_np (inputs) and delta_true_np (true deltas from PDB)
            state_lcg = build_batch_8_15_c(
                state_lcg, N, A_LCG, C_LCG, w, pdb_values,
                BATCH_SIZE, X_np, delta_true_np
            )

            # Move integer inputs to the chosen device.
            X = torch.from_numpy(X_np).to(device=device, dtype=torch.long)

            # For each model in the ensemble, compute quantile-based predictions.
            preds = []
            for model in models:
                logits = model(X)
                preds.append(quantile_class_from_logits(logits, QUANTILE_Q).cpu().numpy())

            # Ensemble combination: take the minimum predicted class index across models
            # for each example (conservative / lower bound).
            delta_pred = np.stack(preds, axis=0).min(axis=0)   # [B]

            # Overestimation of the delta: predicted - ground truth from PDB.
            over = delta_pred.astype(np.int32) - delta_true_np  # delta_pred - delta_true

            # Update global maximum overestimation.
            local_max = int(over.max())
            if local_max > max_over:
                max_over = local_max

            # Collect statistics for strictly positive overestimations.
            pos = over[over > 0]
            if pos.size:
                mx = int(pos.max())
                # Grow histogram if max overestimation exceeds current size.
                if mx >= pos_over_hist.size:
                    new = np.zeros(mx + 1, dtype=np.int64)
                    new[:pos_over_hist.size] = pos_over_hist
                    pos_over_hist = new
                # Update histogram using bincount.
                bc = np.bincount(pos, minlength=pos_over_hist.size)
                pos_over_hist[:bc.size] += bc

            # Periodic progress logging.
            if (it + 1) % 20000 == 0:
                LOG.info("progress=%d/%d  current_max_over=%d  elapsed=%.1fs",
                         it + 1, steps, max_over, time.time() - t0)

    # Final summary: global maximum overestimation across all PDB entries.
    LOG.info("DONE. max_over_delta=%d", max_over)
    LOG.info("positive_over_counts=%s", pos_over_hist[:max_over + 1].tolist())

    # Check whether the LCG returned to its starting state.
    # If true, we likely completed a full cycle over all indices.
    LOG.info("lcg_cycle_closed=%s", str(state_lcg == start_state))

    # Save the maximum overestimation to a small .npy file,
    # which can later be used to construct over-correction tables.
    np.save(f"cert_max_over_q{QUANTILE_Q:.3f}.npy", np.array([max_over], dtype=np.int32))
    LOG.info("saved: cert_max_over_q%.3f.npy", QUANTILE_Q)


if __name__ == "__main__":
    main()
