import os
import time
import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn

from pdb_radix import build_radix_weights
from pdb15_native import unrank_and_manhattan_8_15_c
from pdb_models import PDB8to15Net


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
# Path to the 8–15 pattern database (PDB values for tiles 8..15 + blank).
PDB_PATH = "python/data/pdb_8_15.bin"

# Ensemble of trained delta models for the 8–15 pattern.
# Each model predicts a delta class given the abstract positions.
MODEL_PATHS: List[str] = [
    "models/nn_pdb_8_15_delta_lcg_ens0.pt",
    "models/nn_pdb_8_15_delta_lcg_ens1.pt",
    "models/nn_pdb_8_15_delta_lcg_ens2.pt",
    "models/nn_pdb_8_15_delta_lcg_ens3.pt",
]

# Quantile to use when mapping logits -> class index.
# We then take the elementwise min across ensemble members.
QUANTILE_Q = 0.1

# Number of random PDB states to sample and evaluate.
NUM_SAMPLES = 2_000_000

# Batch size for evaluation.
BATCH_SIZE = 2048

# Model architecture hyperparameters (must match training).
EMBED_DIM = 32
HIDDEN_DIM = 128

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("eval_8_15_admissible")


def quantile_class_from_logits(logits: torch.Tensor, q: float) -> torch.Tensor:
    """
    Map logits to class indices using a given quantile of the predictive distribution.

    Args:
        logits: Tensor of shape [B, C] with raw model outputs.
        q:      Quantile in (0, 1].

    Returns:
        idx: Tensor of shape [B], where for each example:
             idx[b] = #classes with CDF < q, clamped into [0, C - 1].

    Steps:
        1. probs = softmax(logits)
        2. cdf   = cumulative sum of probs over classes
        3. idx   = count of classes where cdf < q
        4. clamp idx to valid class range.
    """
    probs = torch.softmax(logits, dim=1)
    cdf = torch.cumsum(probs, dim=1)
    idx = (cdf < q).sum(dim=1)
    return idx.clamp_(0, logits.size(1) - 1)


def main():
    """
    Random-sampling evaluation of an 8–15 neural delta ensemble
    against the *true* PDB values.

    For NUM_SAMPLES random ranks r in [0, N):
      * Use unrank_and_manhattan_8_15_c(r, w) to:
          - reconstruct the abstract 8–15 sequence (tile positions),
          - get the partial Manhattan distance for tiles 8..15.
      * Read PDB[r] to obtain the exact heuristic value h_true.
      * For each model in the ensemble:
          - run a forward pass to get logits,
          - convert to a quantile-based class index.
      * Combine ensemble predictions by taking the per-sample min delta.
      * Construct h_pred = manhattan_8_15 + delta_pred.
      * Track:
          - over_rate: fraction of samples where h_pred > PDB (inadmissible).
          - max_over: maximum overestimation (h_pred - PDB).
          - mean_gap: average (PDB - h_pred).
          - exact_rate: fraction of samples where h_pred == PDB.
          - mean_h_pred / mean_h_true.

    This script does *not* go through the entire PDB; it evaluates on a large
    random subset (NUM_SAMPLES).
    """
    # Sanity checks for required files.
    for p in MODEL_PATHS:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    if not os.path.exists(PDB_PATH):
        raise FileNotFoundError(PDB_PATH)

    # Load 8–15 PDB into memory (uint8).
    pdb_values = np.fromfile(PDB_PATH, dtype=np.uint8)  # RAM OK
    N = int(pdb_values.shape[0])
    max_pdb = int(pdb_values.max())
    num_classes = max_pdb + 1

    # Build radix weights for partial permutations of length m = 9:
    # tiles 8..15 plus the blank.
    m = 9
    w = build_radix_weights(m)

    # Device selection: use CUDA if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info(
        "device=%s  samples=%d  q=%.3f  N=%s  max_pdb=%d",
        device, NUM_SAMPLES, QUANTILE_Q, f"{N:,}", max_pdb
    )

    # Load ensemble models.
    models: List[nn.Module] = []
    for path in MODEL_PATHS:
        model = PDB8to15Net(EMBED_DIM, HIDDEN_DIM, num_classes)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        models.append(model)

    # RNG for sampling random ranks from [0, N).
    rng = np.random.default_rng(12345)

    # Global stats.
    total = 0
    over_count = 0
    max_over = 0
    sum_gap = 0.0              # sum(PDB - h_pred)  (>= 0 on average if nearly admissible)
    sum_h_pred = 0.0
    sum_h_true = 0.0
    exact_count = 0

    t0 = time.time()

    with torch.no_grad():
        while total < NUM_SAMPLES:
            # Number of samples in this batch.
            bs = min(BATCH_SIZE, NUM_SAMPLES - total)

            # Sample ranks uniformly from [0, N).
            ranks = rng.integers(0, N, size=bs, dtype=np.int64)

            # Arrays for abstract sequences and Manhattan distances.
            seqs = np.empty((bs, m), dtype=np.int64)
            mans = np.empty(bs, dtype=np.int64)

            # For each rank, reconstruct state and partial Manhattan via C++ helper.
            for i, r in enumerate(ranks):
                seq, man = unrank_and_manhattan_8_15_c(int(r), w)
                seqs[i, :] = seq
                mans[i] = man

            # True heuristic values from the PDB.
            pdb_vals = pdb_values[ranks].astype(np.int64)

            # Move inputs to the model device.
            X = torch.from_numpy(seqs).to(device=device, dtype=torch.long)

            # Collect per-model quantile-based predictions.
            preds = []
            for model in models:
                logits = model(X)
                preds.append(quantile_class_from_logits(logits, QUANTILE_Q).cpu().numpy())

            # Ensemble combination: per-sample minimum predicted delta among models.
            delta_pred = np.stack(preds, axis=0).min(axis=0)   # [bs]

            # Predicted and true heuristics.
            h_pred = mans + delta_pred
            h_true = pdb_vals

            # Difference: positive means overestimation (inadmissible).
            diff = h_pred - h_true
            if diff.max() > max_over:
                max_over = int(diff.max())

            over_count += int((diff > 0).sum())
            sum_gap += float((h_true - h_pred).sum())
            sum_h_pred += float(h_pred.sum())
            sum_h_true += float(h_true.sum())
            exact_count += int((h_pred == h_true).sum())

            total += bs

    elapsed = time.time() - t0

    # Aggregate final statistics.
    over_rate = over_count / total
    mean_gap = sum_gap / total
    mean_h_pred = sum_h_pred / total
    mean_h_true = sum_h_true / total
    exact_rate = exact_count / total

    LOG.info("over_rate(h_pred>PDB)=%.8f  max_over=%d", over_rate, max_over)
    LOG.info(
        "mean_h_pred=%.4f  mean_h_true=%.4f  mean_gap(PDB-h_pred)=%.4f",
        mean_h_pred, mean_h_true, mean_gap
    )
    LOG.info("exact_rate(h_pred==PDB)=%.6f", exact_rate)
    LOG.info("time=%.1fs", elapsed)


if __name__ == "__main__":
    main()
