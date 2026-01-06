import os
import time
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

from pdb_radix import build_radix_weights
from pdb15_native import build_batch_8_15_c
from pdb_models import PDB8to15Net


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Full 8–15 pattern database file (PDB values for tiles 8..15 + blank).
PDB_PATH = "python/data/pdb_8_15.bin"

# Paths to the trained ensemble models (same architecture, different seeds).
MODEL_PATHS: List[str] = [
    "models/nn_pdb_8_15_delta_lcg_ens0.pt",
    "models/nn_pdb_8_15_delta_lcg_ens1.pt",
    "models/nn_pdb_8_15_delta_lcg_ens2.pt",
    "models/nn_pdb_8_15_delta_lcg_ens3.pt",
]

# List of quantiles q to evaluate. For each q we:
#   * Convert logits -> class index using that quantile.
#   * Take min across ensemble members (per state).
Q_LIST = [0.3, 0.5, 0.7]

# Full pass settings: we iterate over the entire PDB in batches.
BATCH_SIZE = 2048
STATE0 = 0  # initial LCG state

# LCG params (must match the ones used for training and/or previous certification).
A_LCG = 1201201
C_LCG = 2531021

# Model dimensions (must match training).
EMBED_DIM = 32
HIDDEN_DIM = 128

# Error collection:
# We optionally save:
#   * OVER errors: cases where h_pred > PDB.
#   * HARD errors: underestimates with a large gap (PDB - h_pred >= HARD_GAP_THRESH).
OUT_DIR = "eval_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# For which q values to collect error records (to avoid gigantic files).
COLLECT_QS = {0.5}

SAVE_OVER_ERRORS = True
SAVE_HARD_ERRORS = True
HARD_GAP_THRESH = 2   # "hard" underestimate threshold: PDB - h_pred >= 2

# Progress logging frequency (in steps).
LOG_EVERY_STEPS = 20000


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("eval_full_8_15")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def load_global_cert_k(q: float) -> int | None:
    """
    Try to load a globally certified overestimation bound k for the given quantile q.

    We look for a file named:
        cert_max_over_q{q:0.3f}.npy
    in a few candidate locations. If found, we interpret it as a scalar (or size-1 array)
    and return it as an int. Otherwise we return None.

    The idea:
      * During a separate certification pass you compute:
          k = max_over_delta
        and save it as cert_max_over_q{q}.npy.
      * Here we load k and use it to build an admissible heuristic:
          h_cert = manhattan + max(0, delta_pred - k).
    """
    fname = f"cert_max_over_q{q:0.3f}.npy"
    cand = [
        fname,
        os.path.join("python", fname),
        os.path.join(OUT_DIR, fname),
    ]
    for p in cand:
        if os.path.exists(p):
            v = np.load(p)
            return int(np.array(v).reshape(-1)[0])
    return None


def quantile_class_from_logits(logits: torch.Tensor, q: float) -> torch.Tensor:
    """
    Convert logits to a class index using a quantile of the predictive distribution.

    Args:
        logits: [B, C] tensor of raw network outputs.
        q:      quantile in (0, 1].

    Returns:
        idx: [B] integer tensor of class indices such that:
             idx[b] = number of classes with CDF < q for example b,
             clamped into [0, C-1].
    """
    probs = torch.softmax(logits, dim=1)
    cdf = torch.cumsum(probs, dim=1)
    # idx = count of elements with cdf < q
    idx = (cdf < q).sum(dim=1)
    return idx.clamp_(0, logits.size(1) - 1)


def manhattan_8_15_from_X(X: np.ndarray) -> np.ndarray:
    """
    Compute the Manhattan distance for tiles 8..15 given a packed representation.

    Args:
        X: [B, 9] int32 array.
           X[:, 0..7] are positions (0..15) of tiles 8..15.
           X[:, 8] is the blank position (ignored for this partial Manhattan).

    Returns:
        manhattan: [B] int32 array, where each entry is:
            sum_{t in {8..15}} Manhattan distance of tile t to its goal cell.

    Notes:
        Goal layout is [1..15, 0], so tile t has goal index (t-1).
        For tiles 8..15, goal indices are 7..14.
    """
    pos = X[:, :8].astype(np.int32)   # [B,8]
    r = pos // 4
    c = pos % 4

    # tiles 8..15 goals are indices 7..14 (tile t goal index t-1)
    goal = np.arange(7, 15, dtype=np.int32)  # [7..14]
    gr = goal // 4
    gc = goal % 4

    return (np.abs(r - gr) + np.abs(c - gc)).sum(axis=1).astype(np.int32)


def ensure_counts_len(counts: List[int], need_len: int) -> List[int]:
    """
    Ensure that a list `counts` has at least `need_len` elements,
    extending it with zeros if necessary. Returns the (possibly extended) list.

    Used for growing histograms dynamically.
    """
    if len(counts) < need_len:
        counts.extend([0] * (need_len - len(counts)))
    return counts


def write_error_records(
    fh,
    X_u8: np.ndarray,
    man_u8: np.ndarray,
    dt_u8: np.ndarray,
    dp_u8: np.ndarray,
    gap_i8: np.ndarray,
):
    """
    Write packed error records to an open binary file handle `fh`.

    Each record has the following layout (13 bytes):
        x[9]        : uint8[9]  (positions / sequence indices, 0..15)
        man         : uint8     (Manhattan(8..15))
        delta_true  : uint8     (true delta, e.g. PDB - man)
        delta_pred  : uint8     (predicted delta)
        gap         : int8      (PDB - h_pred or delta_true - delta_pred)

    Args:
        fh      : file-like object opened in "wb" or "ab" mode.
        X_u8    : [B, 9] uint8
        man_u8  : [B] uint8
        dt_u8   : [B] uint8
        dp_u8   : [B] uint8
        gap_i8  : [B] int8
    """
    err_dtype = np.dtype([
        ("x", np.uint8, (9,)),
        ("man", np.uint8),
        ("delta_true", np.uint8),
        ("delta_pred", np.uint8),
        ("gap", np.int8),
    ])
    rec = np.empty(X_u8.shape[0], dtype=err_dtype)
    rec["x"] = X_u8
    rec["man"] = man_u8
    rec["delta_true"] = dt_u8
    rec["delta_pred"] = dp_u8
    rec["gap"] = gap_i8
    rec.tofile(fh)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    """
    Full evaluation over the entire 8–15 PDB.

    For each batch:
        1. Use build_batch_8_15_c to:
             - generate BATCH_SIZE states via LCG over the PDB indices,
             - fill X_np (positions of tiles 8..15 + blank),
             - fill delta_true_np (true delta per state).
        2. Compute manhattan_8_15_from_X(X_np) and h_true = man + delta_true.
        3. For each model in the ensemble:
             - forward pass, get logits,
             - for each q in Q_LIST, compute quantile-based class indices.
        4. For each q:
             - take ensemble-wise min over predicted deltas,
             - compute h_pred, diff = h_pred - h_true,
               gap = h_true - h_pred.
             - accumulate summary stats and histograms.
             - optionally apply globally certified k to compute h_cert and
               collect separate stats for the "certified" version.
             - optionally write error records to disk:
                 * OVER errors (diff > 0)
                 * HARD errors (gap >= HARD_GAP_THRESH).
    At the end, prints per-q stats and whether the LCG cycle closed.
    """
    # Basic file existence checks.
    for p in MODEL_PATHS:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    if not os.path.exists(PDB_PATH):
        raise FileNotFoundError(PDB_PATH)

    # Load full 8–15 PDB into RAM.
    pdb_values = np.fromfile(PDB_PATH, dtype=np.uint8)  # OK on 32GB RAM
    N = int(pdb_values.shape[0])
    max_pdb = int(pdb_values.max())
    num_classes = max_pdb + 1

    if N % BATCH_SIZE != 0:
        raise ValueError(f"N={N} must be divisible by batch={BATCH_SIZE}")
    steps = N // BATCH_SIZE

    # Build radix weights for partial permutations of length m=9
    # (tiles 8..15 plus blank).
    m = 9
    w = build_radix_weights(m)

    # Choose device: CUDA if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOG.info("device=%s  N=%s  steps=%d  batch=%d  max_pdb=%d  classes=%d",
             device, f"{N:,}", steps, BATCH_SIZE, max_pdb, num_classes)
    LOG.info("Q_LIST=%s", Q_LIST)
    LOG.info("collect_errors_for=%s  OUT_DIR=%s", sorted(COLLECT_QS), OUT_DIR)

    # Load all ensemble members.
    models: List[nn.Module] = []
    for path in MODEL_PATHS:
        model = PDB8to15Net(EMBED_DIM, HIDDEN_DIM, num_classes)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        models.append(model)
    LOG.info("ensemble_size=%d", len(models))

    # Allocate per-q statistics struct.
    # For each q we keep:
    #   * counts, means, max over, histogram of positive diffs, etc.
    stats: Dict[float, Dict[str, Any]] = {}
    for q in Q_LIST:
        stats[q] = dict(
            total=0,
            over_count=0,
            max_over=0,
            exact_count=0,

            sum_gap=0.0,
            sum_abs_gap=0.0,

            sum_manhattan=0.0,
            sum_h_pred=0.0,
            sum_h_true=0.0,

            sum_delta_pred=0.0,
            sum_delta_true=0.0,

            # Histogram for positive diff values:
            # index d -> count of (h_pred - h_true) == d.
            positive_over_counts=[0],  # will grow dynamically

            # Certified version accumulators (after subtracting global k).
            cert_k=load_global_cert_k(q),
            cert_sum_gap=0.0,
            cert_sum_abs_gap=0.0,
            cert_sum_h=0.0,
            cert_sum_delta=0.0,
            cert_exact_count=0,
        )

    # Open error files (if enabled) for each q in COLLECT_QS.
    over_fhs: Dict[float, Any] = {}
    hard_fhs: Dict[float, Any] = {}
    if SAVE_OVER_ERRORS:
        for q in Q_LIST:
            if q in COLLECT_QS:
                p = os.path.join(OUT_DIR, f"errors_over_q{q:0.3f}.bin")
                over_fhs[q] = open(p, "wb")
                LOG.info("will save OVER errors for q=%.3f to %s", q, p)
    if SAVE_HARD_ERRORS:
        for q in Q_LIST:
            if q in COLLECT_QS:
                p = os.path.join(OUT_DIR, f"errors_hard_gap{HARD_GAP_THRESH}_q{q:0.3f}.bin")
                hard_fhs[q] = open(p, "wb")
                LOG.info("will save HARD errors for q=%.3f to %s", q, p)

    # Pre-allocate batch buffers.
    X_np = np.empty((BATCH_SIZE, m), dtype=np.int32)
    delta_true_np = np.empty((BATCH_SIZE,), dtype=np.int32)

    state = STATE0
    t0 = time.time()

    try:
        with torch.no_grad():
            for step in range(1, steps + 1):
                # build_batch_8_15_c:
                #   - updates internal LCG state
                #   - fills X_np and delta_true_np with BATCH_SIZE examples.
                state = build_batch_8_15_c(
                    state=state,
                    N=N,
                    a=A_LCG,
                    c=C_LCG,
                    weights=w,
                    pdb_values=pdb_values,
                    batch_size=BATCH_SIZE,
                    X=X_np,
                    y=delta_true_np,
                )

                man = manhattan_8_15_from_X(X_np)              # [B]
                delta_true = delta_true_np.astype(np.int32)    # [B]
                h_true = man + delta_true                      # [B] = PDB

                # Forward pass through all models once per batch.
                X_t = torch.from_numpy(X_np).to(device=device, dtype=torch.long)

                # For each q we maintain the ensemble-wise minimum predicted delta.
                delta_pred_min = {q: np.full((BATCH_SIZE,), 10_000, dtype=np.int16) for q in Q_LIST}

                for model in models:
                    logits = model(X_t)  # [B, C]
                    for q in Q_LIST:
                        idx = quantile_class_from_logits(logits, q=q)
                        idx_np = idx.to(dtype=torch.int16).cpu().numpy()
                        # Ensemble combination: take min across ensemble members.
                        delta_pred_min[q] = np.minimum(delta_pred_min[q], idx_np)

                # Common compact representations for writing error records.
                X_u8_all = X_np.astype(np.uint8, copy=False)
                man_u8_all = np.clip(man, 0, 255).astype(np.uint8, copy=False)
                dt_u8_all = np.clip(delta_true, 0, 255).astype(np.uint8, copy=False)

                # Evaluate and accumulate stats for each q.
                for q in Q_LIST:
                    s = stats[q]
                    dp = delta_pred_min[q].astype(np.int32)    # [B]
                    h_pred = man + dp                         # [B]
                    diff = h_pred - h_true                    # [B] = dp - delta_true
                    gap = h_true - h_pred                     # [B] = delta_true - dp

                    # Update maximum overestimation.
                    dmax = int(diff.max())
                    if dmax > s["max_over"]:
                        s["max_over"] = dmax

                    # Overestimation counts + histogram.
                    over_mask = diff > 0
                    over_pos = diff[over_mask]
                    s["over_count"] += int(over_mask.sum())

                    if over_pos.size > 0:
                        bc = np.bincount(over_pos.astype(np.int32))
                        s["positive_over_counts"] = ensure_counts_len(s["positive_over_counts"], len(bc))
                        for i, v in enumerate(bc):
                            s["positive_over_counts"][i] += int(v)

                    # Exact matches (h_pred == PDB).
                    s["exact_count"] += int((diff == 0).sum())

                    # Sum gaps and absolute gaps (for averages).
                    s["sum_gap"] += float(gap.sum())
                    s["sum_abs_gap"] += float(np.abs(gap).sum())

                    # Means for various components.
                    s["sum_manhattan"] += float(man.sum())
                    s["sum_h_pred"] += float(h_pred.sum())
                    s["sum_h_true"] += float(h_true.sum())
                    s["sum_delta_pred"] += float(dp.sum())
                    s["sum_delta_true"] += float(delta_true.sum())

                    s["total"] += BATCH_SIZE

                    # If we have a global certified k for this q,
                    # build an admissible heuristic:
                    #   delta_cert = max(0, delta_pred - k)
                    #   h_cert = man + delta_cert
                    k = s["cert_k"]
                    if k is not None:
                        dp_cert = np.maximum(0, dp - k)
                        h_cert = man + dp_cert
                        cert_gap = h_true - h_cert
                        s["cert_sum_gap"] += float(cert_gap.sum())
                        s["cert_sum_abs_gap"] += float(np.abs(cert_gap).sum())
                        s["cert_sum_h"] += float(h_cert.sum())
                        s["cert_sum_delta"] += float(dp_cert.sum())
                        s["cert_exact_count"] += int((h_cert == h_true).sum())

                    # Save error records for selected quantiles.
                    if q in COLLECT_QS:
                        dp_u8_all = np.clip(dp, 0, 255).astype(np.uint8, copy=False)
                        gap_i8_all = np.clip(gap, -128, 127).astype(np.int8, copy=False)

                        # OVER errors: h_pred > PDB.
                        if SAVE_OVER_ERRORS and q in over_fhs:
                            if over_mask.any():
                                write_error_records(
                                    over_fhs[q],
                                    X_u8_all[over_mask],
                                    man_u8_all[over_mask],
                                    dt_u8_all[over_mask],
                                    dp_u8_all[over_mask],
                                    gap_i8_all[over_mask],
                                )

                        # HARD errors: PDB - h_pred >= HARD_GAP_THRESH.
                        if SAVE_HARD_ERRORS and q in hard_fhs:
                            hard_mask = gap >= HARD_GAP_THRESH
                            if hard_mask.any():
                                write_error_records(
                                    hard_fhs[q],
                                    X_u8_all[hard_mask],
                                    man_u8_all[hard_mask],
                                    dt_u8_all[hard_mask],
                                    dp_u8_all[hard_mask],
                                    gap_i8_all[hard_mask],
                                )

                if LOG_EVERY_STEPS and (step % LOG_EVERY_STEPS == 0):
                    LOG.info("progress=%d/%d  elapsed=%.1fs", step, steps, time.time() - t0)

    finally:
        # Always close error files, even if an exception occurred.
        for fh in over_fhs.values():
            fh.close()
        for fh in hard_fhs.values():
            fh.close()

    elapsed = time.time() - t0
    lcg_cycle_closed = (state == STATE0)

    # -----------------------------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------------------------
    LOG.info("")
    for q in Q_LIST:
        s = stats[q]
        total = s["total"]

        over_rate = s["over_count"] / total
        exact_rate = s["exact_count"] / total

        mean_manhattan = s["sum_manhattan"] / total
        mean_h_pred = s["sum_h_pred"] / total
        mean_h_true = s["sum_h_true"] / total

        mean_delta_pred = s["sum_delta_pred"] / total
        mean_delta_true = s["sum_delta_true"] / total
        strength = (mean_delta_pred / mean_delta_true) if mean_delta_true > 1e-12 else float("nan")

        mean_gap = s["sum_gap"] / total
        mean_abs_gap = s["sum_abs_gap"] / total

        LOG.info("========== q=%.3f ==========", q)
        LOG.info("over_rate(h_pred>PDB)=%.8f  max_over=%d", over_rate, s["max_over"])
        LOG.info("mean_manhattan=%.4f  mean_h_pred=%.4f  mean_h_true(PDB)=%.4f",
                 mean_manhattan, mean_h_pred, mean_h_true)
        LOG.info("improvement over Manhattan: mean_delta_pred=%.4f  (true mean_delta=%.4f)  strength=%.4f",
                 mean_delta_pred, mean_delta_true, strength)
        LOG.info("distance from PDB: mean_gap(PDB-h_pred)=%.4f  mean_abs_gap=%.4f  exact_rate=%.6f",
                 mean_gap, mean_abs_gap, exact_rate)

        # Show positive over-counts compactly: histogram of (h_pred - PDB) > 0.
        poc = s["positive_over_counts"]
        # Trim trailing zeros for a shorter printout.
        while len(poc) > 0 and poc[-1] == 0:
            poc.pop()
        LOG.info("positive_over_counts=%s", poc if poc else [0])

        # If we had a global cert_k, print stats for the certified heuristic as well.
        if s["cert_k"] is not None:
            k = s["cert_k"]
            cert_mean_h = s["cert_sum_h"] / total
            cert_mean_gap = s["cert_sum_gap"] / total
            cert_mean_abs_gap = s["cert_sum_abs_gap"] / total
            cert_mean_delta = s["cert_sum_delta"] / total
            cert_strength = (cert_mean_delta / mean_delta_true) if mean_delta_true > 1e-12 else float("nan")
            cert_exact_rate = s["cert_exact_count"] / total

            LOG.info("---- certified (global k=%d) ----", k)
            LOG.info("mean_h_cert=%.4f  mean_delta_cert=%.4f  strength=%.4f",
                     cert_mean_h, cert_mean_delta, cert_strength)
            LOG.info("mean_gap(PDB-h_cert)=%.4f  mean_abs_gap=%.4f  exact_rate=%.6f",
                     cert_mean_gap, cert_mean_abs_gap, cert_exact_rate)
        else:
            LOG.info("no cert file found for q=%.3f (expected cert_max_over_q%0.3f.npy)", q, q)

        LOG.info("")

    LOG.info("lcg_cycle_closed=%s", lcg_cycle_closed)
    LOG.info("time=%.1fs", elapsed)


if __name__ == "__main__":
    main()
