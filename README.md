# Parallel_CPU-GPU_Framework_for_Cost-Bounded_DFS

This project solves the 15-puzzle using a **multi-threaded Batch IDA\*** search, with multiple heuristic modes.

Supported modes:

- **PDB-only (CPU)**: admissible Pattern Database heuristic (7/8 split) - no GPU, no batching service.
- **PDB prune + GPU guide**: pruning is still done by PDB (admissible), while a GPU model runs via `NeuralBatchService` for "guide"-style experiments.
- **GPU prune (NN)**: pruning uses a neural heuristic delivered via `NeuralBatchService` (batched on GPU/CPU).
- **DeepCubeA**: pruning uses a TorchScript model compatible with DeepCubeA-style scoring.

> Important
> - All runtime model files should be located **next to the executable** (in the same `bin/` directory).
> - `--board` **overrides** `--korf`.

---

## Runtime files (place next to the executable)

Assuming you run from `.../bin/` and the executable is located there, put the following files in the **same directory**.

### DeepCubeA mode (`--mode=deepcubea`)
- `puzzle15_torchscript.pt`

### NN modes (`--mode=nn` and `--mode=pdb-guide-nn`)
- `nn_pdb_1_7_delta_full_ts.pt`
- `nn_pdb_8_15_delta_lcg_ens0_ts.pt`
- `nn_pdb_8_15_delta_lcg_ens1_ts.pt`
- `nn_pdb_8_15_delta_lcg_ens2_ts.pt`
- `nn_pdb_8_15_delta_lcg_ens3_ts.pt`

If your build uses correction tables (Quantile + corrections), also place:
- `corr_1_7_0.25.bin`
- `corr_8_15_0.25.bin`

### PDB files (PDB modes)
PDB files are searched/created under `--pdb_dir` (default: current working directory):
- `pdb_1_7.bin`
- `pdb_8_15.bin`

---

## Usage

```text
Usage:
  --mode=pdb | pdb-guide-nn | nn | deepcubea
  --korf=N
  --dinit=N
  --worknum=N
  --threads=N
  --batch=N
  --wait_us=N
  --pdb_dir=PATH
  --board=CSV16 (16 numbers, comma-separated)

Example:
  --board=9,1,3,4,2,5,6,8,10,14,7,12,13,11,15,0
```

### Board selection
- If `--board=...` is provided, the program runs **only that board** and ignores `--korf`.
- Otherwise, the program runs the first `--korf=N` Korf instances.

### Threads defaults
 default `--threads=0` (auto / hardware concurrency)

You can always override these defaults with `--threads=N`.

