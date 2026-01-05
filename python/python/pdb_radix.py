import numpy as np
from pdb15_native import unrank_and_manhattan_1_7_c


# Number of board cells in the 4×4 15-puzzle.
N_CELLS = 16


def build_radix_weights(m: int) -> np.ndarray:
    """
    Build radix weights for ranking/unranking partial permutations of board cells.

    This is the Python equivalent of the C++ `build_radix_weights`:
        weights[i] = P(16 - i - 1, m - i - 1)

    where P(n, k) is the number of k-permutations of n elements.

    Conceptually, we encode a sequence of length m (without repetition) drawn
    from 16 cells (0..15) using a mixed-radix system. The `weights` array
    tells us how many states are “skipped” when we increment each position.

    Args:
        m: Length of the abstract sequence (e.g., 8 for tiles 1..7+blank,
           or 9 for tiles 8..15+blank).

    Returns:
        A NumPy array `w` of shape (m,) with dtype uint64, where:
            w[i] = P(16 - i - 1, m - i - 1)
    """
    w = [1] * m
    for i in range(m - 1):
        prod = 1
        for a in range(m - i - 1):
            prod *= (N_CELLS - i - 1 - a)
        w[i] = prod
    w[m - 1] = 1
    return np.array(w, dtype=np.uint64)


def unrank_partial_py(r: int, w: np.ndarray, m: int):
    """
    Python version of partial unranking for 0..15 without repetition.

    Given a rank r in [0, P(16, m)), the radix weights w, and the sequence
    length m, reconstruct the sequence of m distinct cell indices.

    This is the inverse of the (implicit) `rank_partial` function used on
    the C++ side: it walks through a list of available cells and picks the
    correct element for each position based on the quotient and remainder
    with respect to the corresponding radix weight.

    Args:
        r: Integer rank in [0, P(16, m)).
        w: NumPy array of shape (m,) with dtype uint64.
           Usually created via `build_radix_weights(m)`.
        m: Length of the sequence to reconstruct.

    Returns:
        seq: A Python list of length m containing distinct integers in 0..15.
             These represent the positions of the tiles (and blank) for some
             abstract pattern (e.g. tiles 1..7 + blank).
    """
    avail = list(range(N_CELLS))
    seq = []
    for i in range(m):
        block = int(w[i])
        idx = r // block
        r = r % block
        x = avail[idx]
        seq.append(x)
        del avail[idx]
    return seq


def manhattan_1_7_from_seq(seq):
    """
    Compute Manhattan distance for tiles 1..7 only, given an abstract sequence.

    The abstract sequence encodes the positions of tiles 1..7 and the blank
    on a 4×4 board, but without specifying the other tiles explicitly.

    Args:
        seq: list[int] of length 8.
             seq[0..6] = positions (0..15) of tiles 1..7
             seq[7]    = position of the blank (ignored here).

    Returns:
        total: Integer Manhattan distance:
               sum_t |row(pos_t) - row(goal_t)| + |col(pos_t) - col(goal_t)|,
               for t = 1..7.

    Notes:
        The goal layout is assumed to be [1..15, 0], so the goal position
        of tile t is (t - 1).
    """
    total = 0
    for t in range(1, 8):
        pos = seq[t - 1]
        r, c = divmod(pos, 4)
        goal_pos = t - 1
        gr, gc = divmod(goal_pos, 4)
        total += abs(r - gr) + abs(c - gc)
    return total
