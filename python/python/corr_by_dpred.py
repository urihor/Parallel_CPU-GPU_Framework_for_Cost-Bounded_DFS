# python/build_corr_by_dpred_from_errors_over.py
#
# Build a per-class correction table corr[d_pred] from an error log file.
#
# Input:
#   ERR_PATH: "errors_over_q0.500.bin"
#       A binary file containing fixed-size records with fields:
#         - x          : uint8[9]  sequence indices (0..15) for tiles 8..15 + blank
#         - man        : uint8    Manhattan(8..15)
#         - delta_true : uint8    true delta (PDB - man)
#         - delta_pred : uint8    predicted delta (quantile + min-ensemble)
#         - gap        : int8     (delta_true - delta_pred) or (PDB - h_pred),
#                                 depending on how the file was generated.
#
# Output:
#   OUT_PATH: "corr_by_dpred_q0.500.npy"
#       A NumPy array corr of shape [NUM_CLASSES], where:
#         corr[d] = maximum overestimation (delta_pred - delta_true)
#                   observed among all records with predicted class d.
#
# This is then used as an over-correction table to keep the neural heuristic
# admissible by subtracting corr[delta_pred] from the predicted delta.

import os
import numpy as np

ERR_PATH = "eval_outputs/errors_over_q0.500.bin"
OUT_PATH = "eval_outputs/corr_by_dpred_q0.500.npy"

# Number of delta classes (0..40).
NUM_CLASSES = 41  # delta classes 0..40

# Structured dtype describing one error record.
# Total size = 13 bytes:
#   9 bytes  for x
#   1 byte   for man
#   1 byte   for delta_true
#   1 byte   for delta_pred
#   1 byte   for gap
err_dtype = np.dtype([
    ("x", np.uint8, (9,)),     # sequence indices (0..15) for tiles 8..15 + blank (same encoding as saved)
    ("man", np.uint8),         # Manhattan(8..15)
    ("delta_true", np.uint8),  # true delta: PDB - man
    ("delta_pred", np.uint8),  # predicted delta (quantile + min ensemble)
    ("gap", np.int8),          # e.g. (delta_true - delta_pred) or (PDB - h_pred), depending on how you saved it
])  # total = 13 bytes


def main():
    # Sanity check: file size must be an integer multiple of one record.
    sz = os.path.getsize(ERR_PATH)
    assert sz % err_dtype.itemsize == 0, f"bad file size: {sz} not multiple of {err_dtype.itemsize}"
    n = sz // err_dtype.itemsize
    print(f"records={n:,}  bytes={sz:,}  itemsize={err_dtype.itemsize}")

    # You *could* load everything at once:
    #   data = np.fromfile(ERR_PATH, dtype=err_dtype)
    # but for very large files memmap is more robust and memory-efficient.
    mm = np.memmap(ERR_PATH, dtype=err_dtype, mode="r")

    # corr[d] will store the maximum overestimation observed for predicted class d,
    # i.e., max over all records with delta_pred == d of (delta_pred - delta_true).
    corr = np.zeros(NUM_CLASSES, dtype=np.int16)

    # "over" means delta_pred > delta_true, so over = delta_pred - delta_true > 0.
    # We process the file in chunks to limit peak RAM usage.
    CHUNK = 20_000_000  # 20M * 13 ~ 260MB

    for i in range(0, mm.shape[0], CHUNK):
        part = mm[i:i+CHUNK]

        # Extract predicted and true deltas as int16 for safe arithmetic.
        dp = part["delta_pred"].astype(np.int16)
        dt = part["delta_true"].astype(np.int16)
        over = dp - dt  # positive values indicate overestimation

        # For each predicted class d, update corr[d] with the maximum overestimation
        # seen for that class in this chunk.
        for d in np.unique(dp):
            d = int(d)
            m = (dp == d)
            if m.any():
                mx = int(over[m].max())
                if mx > corr[d]:
                    corr[d] = mx

    # Save the correction table as a NumPy .npy file.
    np.save(OUT_PATH, corr)
    print("saved:", OUT_PATH)

    # Print only the entries that are non-zero (i.e., classes that ever overestimated).
    print("nonzero corr:")
    for d in range(NUM_CLASSES):
        if corr[d] != 0:
            print(f"  d={d:2d}  corr={int(corr[d])}")


if __name__ == "__main__":
    main()
