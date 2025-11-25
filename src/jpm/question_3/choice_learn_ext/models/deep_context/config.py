# src/jpm/question_3/choice_learn_ext/models/deep_context/config.py

"""
Configuration object for the DeepHalo (Deep Context-Dependent Choice) model.
This dataclass specifies all architectural and training-related hyperparameters.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class DeepHaloConfig:
    # Embedding dimension for item representations
    d_embed: int = 32

    # Number of phi-interaction heads and stacked halo layers
    n_heads: int = 4
    n_layers: int = 2

    # Residual behavior:
    #   "standard":  phi receives z^{l-1}
    #   "fixed_base": phi receives z^{0}
    residual_variant: str = "standard"

    # Dropout rate for phi MLPs
    dropout: float = 0.0

    # Input mode:
    #   featureless=True  → item_ids are embedded using vocab_size
    #   featureless=False → raw item features X with dimension d_x
    featureless: bool = True
    vocab_size: Optional[int] = None

    # Required if featureless=False (feature-based items)
    d_x: Optional[int] = None