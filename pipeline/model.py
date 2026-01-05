from __future__ import annotations

import torch
import torch.nn as nn


class CausalTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        history_len: int = 10,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, history_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.0,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, _ = x.shape
        emb = self.embedding(x) + self.pos_embed[:, :T, :]
        mask = torch.triu(torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1)
        feat = self.transformer(emb, mask=mask)
        return self.head(feat[:, -1, :])

