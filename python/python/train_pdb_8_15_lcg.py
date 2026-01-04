import os
import time
import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from pdb15_native import build_batch_8_15_c  # C++ batch builder for 8–15 pattern
from pdb_radix import build_radix_weights
from pdb_models import PDB8to15Net


# ---------- Logging setup ----------
logger = logging.getLogger("train_pdb_8_15_lcg")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)


# ---------- LCG + batch builder via C++ ----------

# LCG constants (must match what you use in other scripts/tools).
# These define a (pseudo-)random walk over [0, N) using a linear congruential generator:
#     state_{k+1} = (A_LCG * state_k + C_LCG) mod N
A_LCG = 1201201
C_LCG = 2531021  # if you change this in the C++ / other scripts, update here as well


def build_batch_native(state: int,
                       batch_size: int,
                       w: np.ndarray,
                       m: int,
                       pdb_values: np.ndarray,
                       N: int) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build one training batch by calling the C++ helper `build_batch_8_15_c`.

    This function uses an LCG over [0, N) to generate a sequence of ranks,
    then, inside C++, unpacks each rank into:
      * an abstract sequence for tiles 8..15 + blank, and
      * the true delta target (PDB(8–15) - Manhattan(8..15)).

    Parameters
    ----------
    state : int
        Current LCG state (0 <= state < N). Will be updated by the C++ function.
    batch_size : int
        Number of samples to generate in this batch.
    w : np.ndarray[int64], shape (m,)
        Radix weights for partial permutations (m = 9 for 8–15 pattern).
    m : int
        Number of symbols in the abstract state (tiles 8..15 + blank => 9).
    pdb_values : np.ndarray[uint8], shape (N,)
        8–15 PDB values; pdb_values[r] is the PDB for rank r.
    N : int
        Total number of PDB states (size of pdb_values).

    Returns
    -------
    X_batch : np.ndarray[int32], shape (batch_size, m)
        Abstract sequences; each row:
          X[b, 0..7] = positions (0..15) of tiles 8..15
          X[b, 8]    = position of the blank.
    y_batch : np.ndarray[int32], shape (batch_size,)
        Delta targets for the 8–15 pattern (PDB(8–15) - Manhattan(8..15)),
        possibly clipped in the C++ implementation.
    new_state : int
        Updated LCG state after generating this batch.
    """
    # Use int32 here because the C++ side uses `int` for the arrays.
    X = np.empty((batch_size, m), dtype=np.int32)
    y = np.empty(batch_size, dtype=np.int32)

    new_state = build_batch_8_15_c(
        int(state),
        int(N),
        int(A_LCG),
        int(C_LCG),
        w,             # the wrapper converts to the correct C types
        pdb_values,    # np.uint8, length N
        int(batch_size),
        X,
        y,
    )

    return X, y, int(new_state)


def main():
    """
    Train an ensemble of delta networks for the 8–15 pattern using LCG sampling.

    High-level overview:
      * Load the 8–15 pattern database (PDB_8_15.bin) into RAM.
      * Build radix weights for m=9 (tiles 8..15 + blank).
      * For each ensemble member:
          - Initialize an LCG state (different per member).
          - For each epoch:
              · Use C++ `build_batch_8_15_c` to generate batches:
                    X: abstract sequences, y: delta targets.
              · Train `PDB8to15Net` with cross-entropy on y.
          - Save PyTorch weights (.pt).
          - Export an ONNX model for later inference (e.g., C++/ORT).

    Sampling:
      The LCG in the C++ helper walks through the rank space [0, N) in a
      pseudo-random but deterministic order. With suitable (A, C) it can
      achieve full period (covers all ranks before repeating). The starting
      state is shifted per ensemble member to decorrelate their training
      trajectories.
    """
    # ---------- Load PDB into RAM ----------
    pdb_path = os.path.join("python", "data", "pdb_8_15.bin")
    if not os.path.exists(pdb_path):
        logger.error(f"PDB file not found at {pdb_path}")
        return

    pdb_values = np.fromfile(pdb_path, dtype=np.uint8)
    N = pdb_values.shape[0]

    logger.info(f"Loaded PDB-8-15 from {pdb_path}")
    logger.info(f"N states = {N:,}")

    max_pdb = int(pdb_values.max())
    logger.info(f"Max PDB(8-15) value = {max_pdb}")
    logger.info("Training target: delta = PDB(8-15) - Manhattan(tiles 8..15)")

    # ---------- Radix weights for 8–15 pattern ----------
    # m = 9: positions of tiles 8..15 plus blank.
    m = 9
    w = build_radix_weights(m).astype(np.int64)
    logger.info(f"Radix weights (m = {m}): {w}")

    # ---------- Device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device.type}")

    # ---------- Training config ----------
    batch_size = 1024
    # One “epoch” here is defined as N / batch_size steps:
    # in total ≈ N samples (depending on LCG behavior and modulo).
    steps_per_epoch = N // batch_size
    num_epochs = 1

    logger.info(
        f"Training config: batch_size={batch_size}, steps_per_epoch={steps_per_epoch}, "
        f"samples/epoch ≈ {steps_per_epoch * batch_size:,}"
    )

    # ---------- ENSEMBLE ----------
    # We train an ensemble of networks with identical architecture but different
    # sampling trajectories (different initial LCG state per member).
    ensemble_size = 4
    base_seed = 123456789

    os.makedirs("models", exist_ok=True)

    for member in range(ensemble_size):
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"=== Training ensemble member {member + 1}/{ensemble_size} ===")
        logger.info("=" * 70)

        # Different starting LCG state per ensemble member (shift along the cycle).
        lcg_state = (base_seed + 1000003 * member) % N

        # ----- Model + optimizer for this ensemble member -----
        embed_dim = 32
        hidden_dim = 128
        num_classes = max_pdb + 1  # delta classes 0..max_pdb

        model = PDB8to15Net(embed_dim, hidden_dim, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Optionally also set PyTorch random seed for reproducibility:
        # torch.manual_seed(base_seed + member)

        # ---------- Training loop ----------
        for epoch in range(num_epochs):
            logger.info(f"=== Member {member + 1}: Epoch {epoch + 1}/{num_epochs} ===")
            epoch_start = time.perf_counter()

            epoch_loss_sum = 0.0
            samples_seen = 0

            pbar = tqdm(
                range(steps_per_epoch),
                desc=f"Member {member + 1} | Epoch {epoch + 1}/{num_epochs}",
            )
            for step in pbar:
                # Build one batch via C++ (LCG + unranking + target computation).
                X_np, y_np, lcg_state = build_batch_native(
                    lcg_state, batch_size, w, m, pdb_values, N
                )

                X = torch.from_numpy(X_np).long().to(device)
                y = torch.from_numpy(y_np).long().to(device)

                optimizer.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                loss_val = float(loss.item())
                epoch_loss_sum += loss_val * batch_size
                samples_seen += batch_size

                # Lightweight progress: average loss over seen samples.
                if (step + 1) % 200 == 0:
                    avg_loss = epoch_loss_sum / samples_seen
                    pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")

            epoch_time = time.perf_counter() - epoch_start
            mean_epoch_loss = epoch_loss_sum / samples_seen if samples_seen > 0 else float("nan")
            logger.info(
                f"Member {member + 1}: Epoch {epoch + 1} finished: "
                f"mean_loss={mean_epoch_loss:.4f}, "
                f"samples_seen={samples_seen:,}, time={epoch_time:.1f}s"
            )

        logger.info(f"Full training run finished for member {member + 1}.")

        # ---------- Save weights for this member ----------
        pt_path = os.path.join("models", f"nn_pdb_8_15_delta_lcg_ens{member}.pt")
        torch.save(model.state_dict(), pt_path)
        logger.info(f"[Member {member + 1}] Saved PyTorch weights to {pt_path}")

        # ---------- Export ONNX for this member ----------
        logger.info(f"[Member {member + 1}] Exporting ONNX model...")

        model.eval()
        dummy_input = torch.zeros(1, 9, dtype=torch.long).to(device)
        onnx_path = os.path.join("models", f"nn_pdb_8_15_delta_lcg_ens{member}.onnx")

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
        logger.info(f"[Member {member + 1}] Exported ONNX model to {onnx_path}")


if __name__ == "__main__":
    main()
