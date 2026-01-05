import torch
from torch.utils.data import Dataset
from pdb15_native import unrank_and_manhattan_1_7_c


class PDB1to7Dataset(Dataset):
    """
    PyTorch Dataset for the 1–7 pattern database (PDB) of the 15-puzzle.

    Each sample corresponds to one abstract state of tiles 1..7 + blank,
    represented by a rank into the P(16, 8) space.

    For each sampled rank i:
      * We call `unrank_and_manhattan_1_7_c(i, w)` to obtain:
          - seq: length-8 sequence of positions (0..15) of tiles 1..7 and blank
          - h_manh: Manhattan(1..7) for that abstract state
      * We read `pdb_values[i]` to get the exact heuristic (PDB value).
      * We compute the delta target:
            delta = PDB(1..7) - Manhattan(1..7)
        and clamp it into [0, max_pdb].

    The model will typically learn to predict this delta class from seq.
    """

    def __init__(self, pdb_values, indices, w, m, max_pdb=33):
        """
        Parameters
        ----------
        pdb_values : np.ndarray[uint8]
            Full 1–7 PDB array; pdb_values[i] is the exact PDB value
            for rank i.
        indices : np.ndarray[int]
            Array of ranks (indices into pdb_values) that this dataset
            will iterate over. Can be a random subset or the full range.
        w : np.ndarray[int64], shape (m,)
            Radix weights for partial permutations of length m = 8,
            typically produced by `build_radix_weights(m)`.
        m : int
            Number of symbols in the abstraction (here m=8: tiles 1..7 + blank).
            Kept here for completeness; the value is implied by w.shape[0].
        max_pdb : int, optional
            Upper bound used to clip the delta target; any delta > max_pdb
            is set to max_pdb. This limits the number of output classes.
        """
        self.pdb_values = pdb_values
        self.indices = indices
        self.w = w
        self.m = m
        self.max_pdb = max_pdb

    def __len__(self):
        """Return the number of samples (ranks) in this dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        Return the (input, target) pair for sample `idx`.

        Returns
        -------
        x : torch.LongTensor, shape [8]
            Abstract sequence:
              x[0..6] = positions (0..15) of tiles 1..7
              x[7]    = position of the blank.
        y : torch.LongTensor, scalar
            Delta class:
                y = clipped(PDB(1..7) - Manhattan(1..7))
            where the clipping is into [0, max_pdb].
        """
        # Take the rank from the indices array
        i = int(self.indices[idx])

        # Call the C++ helper and get the *seq* representation (not a full board state).
        seq, h_manh = unrank_and_manhattan_1_7_c(i, self.w)

        # Convert seq to a torch tensor of shape [8]
        x = torch.tensor(seq, dtype=torch.long)

        # Compute delta = PDB(1–7) - Manhattan(1..7)
        pdb_val = int(self.pdb_values[i])
        delta = pdb_val - h_manh

        # Clamp into the valid class range [0, max_pdb]
        if delta < 0:
            delta = 0
        elif delta > self.max_pdb:
            delta = self.max_pdb

        y = torch.tensor(delta, dtype=torch.long)

        return x, y
