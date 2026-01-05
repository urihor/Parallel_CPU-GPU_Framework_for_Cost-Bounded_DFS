# export_ts_from_state_dict.py
#
# Utility script to convert trained PyTorch models (saved as state_dict .pt files)
# into TorchScript .pt files that can be loaded from C++ (libtorch).
#
# It defines two architectures:
#   - PDB1to7Net  : for the 1–7 pattern (input shape [B, 8])
#   - PDB8to15Net : for the 8–15 pattern (input shape [B, 9])
#
# And two helpers:
#   - export_1_7(...)   : load state_dict for 1–7 network and export to TorchScript
#   - export_8_15(...)  : load state_dict for 8–15 network and export to TorchScript
#
# In __main__ it exports:
#   * models/nn_pdb_1_7_delta_full.pt           -> models/nn_pdb_1_7_delta_full_ts.pt
#   * models/nn_pdb_8_15_delta_lcg_ens{i}.pt    -> models/nn_pdb_8_15_delta_lcg_ens{i}_ts.pt

import torch
import torch.nn as nn


class PDB1to7Net(nn.Module):
    """
    Simple feed-forward network for the 1–7 pattern.

    Input:
        x: LongTensor of shape [B, 8], where each entry is a cell index (0..15)
           for tiles 1..7 and the blank (or any fixed encoding used in training).

    Architecture:
        - Embedding(16, embed_dim)
        - Flatten 8 embeddings into a single vector of size 8 * embed_dim
        - Linear -> ReLU -> Linear -> ReLU -> Linear (to num_classes)
    """

    def __init__(self, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(16, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        e = self.embed(x)              # [B,8,E]
        e = e.view(e.size(0), -1)      # [B,8E]
        h = self.act(self.fc1(e))
        h = self.act(self.fc2(h))
        return self.out(h)             # [B,C]


class PDB8to15Net(nn.Module):
    """
    Simple feed-forward network for the 8–15 pattern.

    Input:
        x: LongTensor of shape [B, 9], typically positions of tiles 8..15
           and the blank (or a similar encoding).

    Architecture:
        - Embedding(16, embed_dim)
        - Flatten 9 embeddings into a single vector of size 9 * embed_dim
        - Linear -> ReLU -> Linear (to num_classes)
    """

    def __init__(self, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(16, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 9, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        e = self.embed(x)              # [B,9,E]
        e = e.view(e.size(0), -1)      # [B,9E]
        h = torch.relu(self.fc1(e))
        return self.fc2(h)             # [B,C]


def export_1_7(sd_path, out_path):
    """
    Load a trained 1–7 model from a state_dict file and export it as TorchScript.

    Parameters:
        sd_path  - path to the .pt file containing the state_dict
        out_path - path to the output TorchScript .pt file

    The network dimensions (embed_dim, hidden_dim, num_classes) are inferred
    from the shapes of the loaded parameters, so they don't need to be hard-coded.
    """
    sd = torch.load(sd_path, map_location="cpu")
    E = sd["embed.weight"].shape[1]
    H = sd["fc1.weight"].shape[0]
    C = sd["out.weight"].shape[0]
    m = PDB1to7Net(E, H, C)
    m.load_state_dict(sd)
    m.eval()

    # Example input: batch size 1, sequence length 8 (positions).
    example = torch.zeros(1, 8, dtype=torch.long)

    # Trace the model with the example input to obtain a TorchScript module.
    ts = torch.jit.trace(m, example)
    ts.save(out_path)
    print("saved", out_path)


def export_8_15(sd_path, out_path):
    """
    Load a trained 8–15 model from a state_dict file and export it as TorchScript.

    Parameters:
        sd_path  - path to the .pt file containing the state_dict
        out_path - path to the output TorchScript .pt file

    As above, the architecture dimensions are inferred from parameter shapes.
    """
    sd = torch.load(sd_path, map_location="cpu")
    E = sd["embed.weight"].shape[1]
    H = sd["fc1.weight"].shape[0]
    C = sd["fc2.weight"].shape[0]
    m = PDB8to15Net(E, H, C)
    m.load_state_dict(sd)
    m.eval()

    # Example input: batch size 1, sequence length 9.
    example = torch.zeros(1, 9, dtype=torch.long)

    # Trace to TorchScript.
    ts = torch.jit.trace(m, example)
    ts.save(out_path)
    print("saved", out_path)


if __name__ == "__main__":
    # Export the single 1–7 model.
    export_1_7("models/nn_pdb_1_7_delta_full.pt",
               "models/nn_pdb_1_7_delta_full_ts.pt")

    # Export the 8–15 ensemble models (e.g., 4 members).
    for i in range(4):
        export_8_15(f"models/nn_pdb_8_15_delta_lcg_ens{i}.pt",
                    f"models/nn_pdb_8_15_delta_lcg_ens{i}_ts.pt")
