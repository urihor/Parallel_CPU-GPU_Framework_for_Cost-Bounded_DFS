import torch.nn as nn
import torch


class PDB1to7Net(nn.Module):
    """
    Feed-forward network for the 1–7 pattern delta prediction.

    Input:
        x: LongTensor of shape [B, 8]
           Each row encodes one abstract state:
             x[b, 0..6] = positions (0..15) of tiles 1..7
             x[b, 7]    = position of the blank.

    Architecture:
        - Embedding layer over 16 possible board cells (0..15),
          applied to each of the 8 positions.
        - Flatten the 8 embeddings into a single feature vector.
        - Two fully-connected hidden layers with ReLU activations.
        - Final linear layer outputs logits over delta classes 0..(num_classes-1),
          where the target is typically:
              delta = PDB(1..7) - Manhattan(1..7),
          optionally clipped to some max value.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_classes: int):
        """
        Parameters
        ----------
        embed_dim : int
            Dimension of the cell-embedding vectors.
        hidden_dim : int
            Width of the hidden fully-connected layers.
        num_classes : int
            Number of delta classes (e.g. max_delta + 1).
        """
        super().__init__()
        # 16 possible board cells, each mapped to an `embed_dim`-dim vector.
        self.embed = nn.Embedding(num_embeddings=16, embedding_dim=embed_dim)
        # Input to the first FC layer is embeddings for 8 positions, flattened.
        self.fc1 = nn.Linear(embed_dim * 8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: LongTensor of shape [B, 8] with values in 0..15
               (cell indices for the 8 symbols: tiles 1..7 + blank).

        Returns:
            logits: FloatTensor of shape [B, num_classes],
                    unnormalized scores over delta classes.
        """
        # x: [B, 8]
        emb = self.embed(x)               # [B, 8, E]
        emb = emb.view(emb.size(0), -1)   # [B, 8 * E]
        h = self.act(self.fc1(emb))
        h = self.act(self.fc2(h))
        logits = self.out(h)              # [B, num_classes]
        return logits


class PDB8to15Net(nn.Module):
    """
    Feed-forward network for the 8–15 pattern delta prediction.

    Input:
        x: LongTensor of shape [B, 9]
           Each row encodes one abstract state:
             x[b, 0..7] = positions (0..15) of tiles 8..15
             x[b, 8]    = position of the blank.

    Architecture:
        - Embedding over 16 board cells (0..15) applied to 9 positions.
        - Flatten the 9 embeddings.
        - One hidden fully-connected layer with ReLU.
        - Final linear layer outputs logits over delta classes.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_classes: int):
        """
        Parameters
        ----------
        embed_dim : int
            Dimension of the cell-embedding vectors.
        hidden_dim : int
            Width of the hidden fully-connected layer.
        num_classes : int
            Number of delta classes (e.g. max_delta + 1).
        """
        super().__init__()
        # 16 possible board cells, each with an embedding vector.
        self.embed = nn.Embedding(16, embed_dim)  # 16 board cells
        # 9 positions (tiles 8..15 + blank), each mapped to an embedding.
        self.fc1 = nn.Linear(embed_dim * 9, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: LongTensor of shape [B, 9]
               with cell indices 0..15.

        Returns:
            logits: FloatTensor of shape [B, num_classes],
                    unnormalized scores over delta classes.
        """
        # x: [B, 9]
        e = self.embed(x)              # [B, 9, embed_dim]
        e = e.view(e.size(0), -1)      # [B, 9 * embed_dim]
        h = torch.relu(self.fc1(e))
        logits = self.fc2(h)
        return logits
