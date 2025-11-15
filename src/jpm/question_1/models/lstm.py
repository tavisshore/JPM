import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers, optimizers

from ..data.ed import EdgarDataLoader
from .losses import EnforceBalance, bs_loss


def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--ticker", type=str, default="AAPL")
    p.add_argument("--cache_dir", type=str, default="/Users/tavisshore/Desktop/HK/data")

    # Model params
    p.add_argument("--hidden_units", type=int, default=128)
    p.add_argument("--dense_units", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)

    # BS identiy loss
    p.add_argument("--lambda_balance", type=float, default=1e-4)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data = EdgarDataLoader(ticker=args.ticker, cache_dir=args.cache_dir)

    # Model definition
    inputs = Input(
        shape=(None, data.num_features),
        dtype=tf.float32,
        name="inputs",
    )

    x = layers.LSTM(args.hidden_units, return_sequences=False, name="lstm_1")(inputs)
    x = layers.Dense(args.dense_units, activation="relu", name="dense_1")(x)
    outputs_raw = layers.Dense(data.num_targets, name="next_quarter")(x)

    outputs = EnforceBalance(
        feature_mappings=data.feature_mappings,
        slack_name="accumulated_other_comprehensive_income_loss_net_of_tax",
        feature_names=data.bs_keys,
    )(outputs_raw)

    model = Model(inputs=inputs, outputs=outputs)

    loss_fn = bs_loss(
        feature_means=data.feat_stat["mean"][data.tgt_indices],
        feature_stds=data.feat_stat["std"][data.tgt_indices],
        feature_mappings=data.feature_mappings,
        lambda_balance=args.lambda_balance,
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr), loss=loss_fn, metrics=["mae"]
    )
    model.summary()

    model.fit(data.train_dataset, epochs=args.epochs)

    y_pred = model.predict(data.train_dataset)
    y_true = np.concatenate([y for x, y in data.train_dataset], axis=0)

    # Test on withheld block
    y_test = model.predict(data.test_dataset)
    # Now build the actual balance sheet from this?
