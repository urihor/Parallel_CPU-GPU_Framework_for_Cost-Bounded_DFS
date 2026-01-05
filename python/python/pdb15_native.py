import os
import ctypes
import numpy as np

# -------------------------------------------------------------------
# Locate and load the native shared library
# -------------------------------------------------------------------
# We assume the C++ code was built as libpdb15_native.so and placed
# under one of a few possible build directories relative to the
# project root. This allows using the same Python module both under
# WSL and regular local builds.

# Project paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

# Try several possible build dirs (WSL / local)
_candidate_libs = [
    os.path.join(_PROJECT_ROOT, "build-wsl", "libpdb15_native.so"),
    os.path.join(_PROJECT_ROOT, "build", "libpdb15_native.so"),
    os.path.join(_PROJECT_ROOT, "cmake-build-debug", "libpdb15_native.so"),
]

_LIB_PATH = None
for p in _candidate_libs:
    if os.path.exists(p):
        _LIB_PATH = p
        break

if _LIB_PATH is None:
    raise FileNotFoundError(f"Shared library libpdb15_native.so not found in: {_candidate_libs}")

# Load the shared library via ctypes.
_lib = ctypes.CDLL(_LIB_PATH)

# -------------------------------------------------------------------
# C function signatures
# -------------------------------------------------------------------
# Here we declare the ctypes signatures for the functions exported
# by libpdb15_native.so. This must match the C++ declarations exactly.

# void unrank_and_manhattan_1_7(uint64_t r,
#                               const uint64_t *weights,
#                               int m,
#                               int *out_seq,
#                               int *out_manhattan);
_lib.unrank_and_manhattan_1_7.argtypes = [
    ctypes.c_uint64,                 # r
    ctypes.POINTER(ctypes.c_uint64), # weights
    ctypes.c_int,                    # m
    ctypes.POINTER(ctypes.c_int),    # out_seq
    ctypes.POINTER(ctypes.c_int),    # out_manhattan
]
_lib.unrank_and_manhattan_1_7.restype = None

# void unrank_and_manhattan_8_15(uint64_t r,
#                                const uint64_t *weights,
#                                int m,
#                                int *out_seq,
#                                int *out_manhattan);
_lib.unrank_and_manhattan_8_15.argtypes = [
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_lib.unrank_and_manhattan_8_15.restype = None

# uint64_t build_batch_8_15(uint64_t state,
#                           uint64_t N,
#                           uint64_t a,
#                           uint64_t c,
#                           const uint64_t *weights,
#                           int m,
#                           const uint8_t *pdb_values,
#                           int batch_size,
#                           int *out_seq,   // flat: batch_size * m
#                           int *out_delta);
_lib.build_batch_8_15.argtypes = [
    ctypes.c_uint64,                  # state
    ctypes.c_uint64,                  # N
    ctypes.c_uint64,                  # a
    ctypes.c_uint64,                  # c
    ctypes.POINTER(ctypes.c_uint64),  # weights
    ctypes.c_int,                     # m
    ctypes.POINTER(ctypes.c_uint8),   # pdb_values
    ctypes.c_int,                     # batch_size
    ctypes.POINTER(ctypes.c_int),     # out_seq (flat: batch_size * m)
    ctypes.POINTER(ctypes.c_int),     # out_delta
]
_lib.build_batch_8_15.restype = ctypes.c_uint64

# uint64_t build_batch_1_7(uint64_t state,
#                          uint64_t N,
#                          uint64_t a,
#                          uint64_t c,
#                          const uint64_t *weights,
#                          int m,
#                          const uint8_t *pdb_values,
#                          int batch_size,
#                          int *out_seq,
#                          int *out_delta);
_lib.build_batch_1_7.argtypes = [
    ctypes.c_uint64,                  # state
    ctypes.c_uint64,                  # N
    ctypes.c_uint64,                  # a
    ctypes.c_uint64,                  # c
    ctypes.POINTER(ctypes.c_uint64),  # weights
    ctypes.c_int,                     # m
    ctypes.POINTER(ctypes.c_uint8),   # pdb_values
    ctypes.c_int,                     # batch_size
    ctypes.POINTER(ctypes.c_int),     # out_seq
    ctypes.POINTER(ctypes.c_int),     # out_delta
]
_lib.build_batch_1_7.restype = ctypes.c_uint64


# -------------------------------------------------------------------
# Python wrappers
# -------------------------------------------------------------------

def unrank_and_manhattan_1_7_c(r: int, weights: np.ndarray) -> tuple[list[int], int]:
    """
    Python wrapper around the native C++ `unrank_and_manhattan_1_7`.

    Parameters
    ----------
    r : int
        Rank in [0, P(16, 8)).
        This identifies one abstract state for tiles 1..7 + blank,
        using the partial permutation ranking scheme.
    weights : np.ndarray[int64], shape (8,)
        Radix weights P(16 - i - 1, m - i - 1) for m = 8.
        Typically produced by `build_radix_weights(8)`.

    Returns
    -------
    seq : list[int]
        Length-8 sequence of positions (0..15):
            seq[0..6] = positions of tiles 1..7,
            seq[7]    = position of the blank.
    manhattan : int
        Manhattan distance of tiles 1..7 to their goal cells.
    """
    m = int(weights.shape[0])
    w_c = (ctypes.c_uint64 * m)()
    for i in range(m):
        w_c[i] = ctypes.c_uint64(int(weights[i]))

    out_seq = (ctypes.c_int * m)()
    out_manh = ctypes.c_int()

    _lib.unrank_and_manhattan_1_7(
        ctypes.c_uint64(r),
        w_c,
        ctypes.c_int(m),
        out_seq,
        ctypes.byref(out_manh),
    )

    seq = [int(out_seq[i]) for i in range(m)]
    manh = int(out_manh.value)
    return seq, manh


def unrank_and_manhattan_8_15_c(r: int, weights: np.ndarray) -> tuple[list[int], int]:
    """
    Python wrapper around the native C++ `unrank_and_manhattan_8_15`.

    Parameters
    ----------
    r : int
        Rank in [0, P(16, 9)).
        This identifies one abstract state for tiles 8..15 + blank.
    weights : np.ndarray[int64], shape (9,)
        Radix weights for m = 9 (tiles 8..15 plus blank).

    Returns
    -------
    seq : list[int]
        Length-9 sequence of positions (0..15):
            seq[0..7] = positions of tiles 8..15,
            seq[8]    = position of the blank.
    manhattan : int
        Manhattan distance of tiles 8..15 to their goal cells.
    """
    m = int(weights.shape[0])
    w_c = (ctypes.c_uint64 * m)()
    for i in range(m):
        w_c[i] = ctypes.c_uint64(int(weights[i]))

    out_seq = (ctypes.c_int * m)()
    out_manh = ctypes.c_int()

    _lib.unrank_and_manhattan_8_15(
        ctypes.c_uint64(r),
        w_c,
        ctypes.c_int(m),
        out_seq,
        ctypes.byref(out_manh),
    )

    seq = [int(out_seq[i]) for i in range(m)]
    manh = int(out_manh.value)
    return seq, manh


def build_batch_8_15_c(state: int,
                       N: int,
                       a: int,
                       c: int,
                       weights: np.ndarray,
                       pdb_values: np.ndarray,
                       batch_size: int,
                       X: np.ndarray,
                       y: np.ndarray) -> int:
    """
    Python wrapper for the native C++ `build_batch_8_15`.

    This function uses an LCG over [0, N) to generate `batch_size` abstract
    states for the 8..15 pattern, then fills:
      * `X` with the corresponding sequences, and
      * `y` with the true delta values (PDB - Manhattan(8..15)).

    The LCG state is updated and the new state is returned.

    Parameters
    ----------
    state : int
        Current LCG state (seed). Returned LCG state can be reused for
        the next call to continue the sequence.
    N : int
        Number of entries in the 8–15 PDB (pdb_values length).
    a : int
        LCG multiplier.
    c : int
        LCG increment.
    weights : np.ndarray[int64], shape (m,)
        Radix weights for partial permutations of length m = 9.
    pdb_values : np.ndarray[uint8], shape (N,)
        The full 8–15 PDB array, where pdb_values[r] is the exact heuristic
        for rank r.
    batch_size : int
        Number of samples to generate in this call.
    X : np.ndarray[int32], shape (batch_size, m)
        Output buffer for sequences. On return:
          X[b, 0..7] = positions of tiles 8..15
          X[b, 8]    = blank position.
    y : np.ndarray[int32], shape (batch_size,)
        Output buffer for deltas. On return:
          y[b] = PDB - Manhattan(8..15) for X[b].

    Returns
    -------
    new_state : int
        Updated LCG state after generating the batch.
    """
    m = int(weights.shape[0])

    w_c = (ctypes.c_uint64 * m)()
    for i in range(m):
        w_c[i] = ctypes.c_uint64(int(weights[i]))

    assert X.shape == (batch_size, m)
    assert X.dtype == np.int32
    assert y.shape == (batch_size,)
    assert y.dtype == np.int32
    assert pdb_values.dtype == np.uint8

    pdb_c = pdb_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    X_c = X.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    y_c = y.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    new_state = _lib.build_batch_8_15(
        ctypes.c_uint64(state),
        ctypes.c_uint64(N),
        ctypes.c_uint64(a),
        ctypes.c_uint64(c),
        w_c,
        ctypes.c_int(m),
        pdb_c,
        ctypes.c_int(batch_size),
        X_c,
        y_c,
    )

    return int(new_state)


def build_batch_1_7_c(state: int,
                      N: int,
                      a: int,
                      c: int,
                      weights: np.ndarray,
                      pdb_values: np.ndarray,
                      batch_size: int,
                      X: np.ndarray,
                      y: np.ndarray) -> int:
    """
    Python wrapper for the native C++ `build_batch_1_7`.

    Similar to `build_batch_8_15_c`, but for the 1..7 pattern.

    This function uses an LCG over [0, N) to generate `batch_size` abstract
    states for tiles 1..7 + blank, then fills:
      * `X` with the corresponding sequences, and
      * `y` with the true delta values (PDB - Manhattan(1..7)).

    Parameters
    ----------
    state : int
        Current LCG state (seed). Returned LCG state can be reused for
        the next call to continue the sequence.
    N : int
        Number of entries in the 1–7 PDB (pdb_values length).
    a : int
        LCG multiplier.
    c : int
        LCG increment.
    weights : np.ndarray[int64], shape (m,)
        Radix weights for partial permutations of length m = 8.
    pdb_values : np.ndarray[uint8], shape (N,)
        The full 1–7 PDB array.
    batch_size : int
        Number of samples to generate in this call.
    X : np.ndarray[int32], shape (batch_size, m)
        Output buffer for sequences. On return:
          X[b, 0..6] = positions of tiles 1..7
          X[b, 7]    = blank position.
    y : np.ndarray[int32], shape (batch_size,)
        Output buffer for deltas. On return:
          y[b] = PDB - Manhattan(1..7) for X[b].

    Returns
    -------
    new_state : int
        Updated LCG state after generating the batch.
    """
    m = int(weights.shape[0])

    w_c = (ctypes.c_uint64 * m)()
    for i in range(m):
        w_c[i] = ctypes.c_uint64(int(weights[i]))

    assert X.shape == (batch_size, m)
    assert X.dtype == np.int32
    assert y.shape == (batch_size,)
    assert y.dtype == np.int32
    assert pdb_values.dtype == np.uint8

    pdb_c = pdb_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    X_c = X.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    y_c = y.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    new_state = _lib.build_batch_1_7(
        ctypes.c_uint64(state),
        ctypes.c_uint64(N),
        ctypes.c_uint64(a),
        ctypes.c_uint64(c),
        w_c,
        ctypes.c_int(m),
        pdb_c,
        ctypes.c_int(batch_size),
        X_c,
        y_c,
    )
    return int(new_state)
