# src/jpm/question_3/choice_learn_ext/models/deep_context/main_model.py
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from .deep_halo_core import DeepContextChoiceModel
from .trainer import Trainer


class DeepHaloChoiceModel:
    """
    Public-facing wrapper that follows the choice-learn estimator API.
    """

    def __init__(
        self,
        num_items,
        lr=1e-3,
        epochs=30,
        batch_size=128,
        d_embed=16,
        n_blocks=2,
        n_heads=2,
        residual_variant="fixed_base",
        verbose=1,
    ):
        self.num_items = num_items
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        # Underlying TF model
        self.model = DeepContextChoiceModel(
            num_items=num_items,
            d_embed=d_embed,
            n_blocks=n_blocks,
            n_heads=n_heads,
            residual_variant=residual_variant,
        )

        self.trainer = Trainer(self.model, lr=lr)

    # ------------------------------------------------------------------
    # Data conversion utilities
    # ------------------------------------------------------------------

    def _df_to_tensors(self, df: pd.DataFrame):
        """
        Expected df columns:
            - 'available' : list of ints or list of 0/1
            - 'item_ids'  : list of ints
            - 'choice'    : int (chosen item)
        """

        available = np.stack(df["available"].values).astype(np.float32)
        item_ids = np.stack(df["item_ids"].values).astype(np.int32)
        choices = df["choice"].values.astype(np.int32)

        return (
            tf.convert_to_tensor(available),
            tf.convert_to_tensor(item_ids),
            tf.convert_to_tensor(choices),
        )

    # ------------------------------------------------------------------
    # Training API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame):
        available, item_ids, choices = self._df_to_tensors(df)

        self.trainer.fit_arrays(
            available=available,
            item_ids=item_ids,
            choices=choices,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
        )

        return self

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame):
        available, item_ids, _ = self._df_to_tensors(df)

        out = self.model(
            {"available": available, "item_ids": item_ids},
            training=False,
        )
        return np.exp(out["log_probs"].numpy())

    def predict(self, df: pd.DataFrame):
        probs = self.predict_proba(df)
        return probs.argmax(axis=1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def log_likelihood(self, df: pd.DataFrame):
        available, item_ids, choices = self._df_to_tensors(df)
        return float(self.model.nll(
            {"available": available, "item_ids": item_ids, "choice": choices},
            training=False,
        ).numpy())

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path):
        config = {
            "num_items": self.num_items,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }
        with open(path + ".json", "w") as f:
            json.dump(config, f)

        self.model.save_weights(path + "_weights.h5")

    @classmethod
    def load(cls, path):
        with open(path + ".json", "r") as f:
            config = json.load(f)

        model = cls(**config)
        model.model.load_weights(path + "_weights.h5")
        return model
