"""Optional temporal fusion model for modality embeddings or logits.

This module is intentionally separate from the scikit-learn phase-two models.
It requires PyTorch only when instantiated, so the current tabular pipeline
continues to work in environments without a GPU or a PyTorch installation.
"""

from __future__ import annotations

from typing import Mapping, Sequence

try:
    from torch import nn as _torch_nn
except ImportError:  # pragma: no cover - environment dependent
    _torch_nn = None

_TorchModule = _torch_nn.Module if _torch_nn is not None else object


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "pytorch_required_for_lstm_fusion; install the training extra before use"
        ) from error
    return torch, nn


class LSTMFusionModel(_TorchModule):
    """Fuse modality representations across consecutive windows.

    Input tensors contain one concatenated representation per timestep. The
    caller must append one availability mask value per modality; this lets the
    model distinguish a real zero from a missing modality.
    """

    def __init__(
        self,
        modality_dimensions: Mapping[str, int],
        label_count: int,
        *,
        hidden_size: int = 64,
        layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        torch, nn = _torch()
        super().__init__()
        if not modality_dimensions or any(value <= 0 for value in modality_dimensions.values()):
            raise ValueError("modality_dimensions_must_be_positive")
        if label_count <= 0 or hidden_size <= 0 or layers <= 0:
            raise ValueError("lstm_dimensions_must_be_positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout_must_be_in_range")
        self.modality_dimensions = dict(modality_dimensions)
        self.label_count = label_count
        self.input_size = sum(modality_dimensions.values()) + len(modality_dimensions)
        self.network = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, label_count)

    def forward(self, sequences, lengths: Sequence[int] | None = None):
        """Return logits for one multilabel output per sequence."""
        torch, nn = _torch()
        if sequences.ndim != 3 or sequences.shape[-1] != self.input_size:
            raise ValueError("invalid_lstm_sequence_shape")
        if lengths is None:
            _, state = self.network(sequences)
            final_state = state[0][-1]
        else:
            if len(lengths) != sequences.shape[0] or any(
                length <= 0 or length > sequences.shape[1] for length in lengths
            ):
                raise ValueError("invalid_lstm_sequence_lengths")
            packed = nn.utils.rnn.pack_padded_sequence(
                sequences, torch.as_tensor(lengths, device=sequences.device),
                batch_first=True, enforce_sorted=False,
            )
            _, state = self.network(packed)
            final_state = state[0][-1]
        return self.classifier(final_state)

    def probabilities(self, sequences, lengths: Sequence[int] | None = None):
        torch, _ = _torch()
        return torch.sigmoid(self.forward(sequences, lengths))
