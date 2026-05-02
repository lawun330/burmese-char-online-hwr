from __future__ import annotations

import torch
from torch import nn


class RnnCtc(nn.Module):
    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.in_proj = nn.Linear(input_size, hidden_size)
        self.rnn = nn.RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, vocab_size + 1)  # + blank at 0

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        lengths: (B,)
        returns log_probs: (T, B, C) for CTCLoss
        """
        x = self.in_proj(x)
        x = torch.relu(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.head(out)  # (B, T, C)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, C)
        return log_probs

