import tensorflow as tf
from tensorflow.keras.layers import Layer


def bs_loss(
    feature_means,
    feature_stds,
    feature_mappings,
    lambda_balance=1e-4,
):
    means_tf = tf.constant(feature_means, dtype=tf.float32)
    stds_tf = tf.constant(feature_stds, dtype=tf.float32)

    asset_idx_tf = tf.constant(feature_mappings["assets"], dtype=tf.int32)
    liability_idx_tf = tf.constant(feature_mappings["liabilities"], dtype=tf.int32)
    equity_idx_tf = tf.constant(feature_mappings["equity"], dtype=tf.int32)

    def loss(y_true, y_pred):
        # base loss = MSE in scaled space
        base_mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # Unscale predictions
        y_pred_unscaled = y_pred * stds_tf + means_tf  # (batch, F)

        # Compute A, L & E sums in unscaled space
        assets = tf.reduce_sum(
            tf.gather(y_pred_unscaled, asset_idx_tf, axis=-1), axis=-1
        )
        liabilities = tf.reduce_sum(
            tf.gather(y_pred_unscaled, liability_idx_tf, axis=-1), axis=-1
        )
        equity = tf.reduce_sum(
            tf.gather(y_pred_unscaled, equity_idx_tf, axis=-1), axis=-1
        )

        # Asses conformity to A = L + E
        violation = assets - (liabilities + equity)
        identity_penalty = tf.reduce_mean(tf.square(violation))

        return base_mse + lambda_balance * identity_penalty

    return loss


class EnforceBalance(Layer):
    def __init__(
        self,
        feature_mappings,
        slack_name="accumulated_other_comprehensive_income_loss_net_of_tax",
        feature_names=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.asset_idx = tf.constant(feature_mappings["assets"], dtype=tf.int32)
        self.liability_idx = tf.constant(
            feature_mappings["liabilities"], dtype=tf.int32
        )
        self.equity_idx = tf.constant(feature_mappings["equity"], dtype=tf.int32)

        if feature_names is None:
            raise ValueError("feature_names required to find slack index")

        assert (
            slack_name in feature_names
        ), f"Slack variable '{slack_name}' not found in feature names"
        self.slack_idx = feature_names.index(slack_name)

    def call(self, y):
        assets = tf.reduce_sum(
            tf.gather(y, self.asset_idx, axis=-1), axis=-1, keepdims=True
        )
        liabilities = tf.reduce_sum(
            tf.gather(y, self.liability_idx, axis=-1), axis=-1, keepdims=True
        )
        equity = tf.reduce_sum(
            tf.gather(y, self.equity_idx, axis=-1), axis=-1, keepdims=True
        )
        diff = assets - (liabilities + equity)

        # Build correction tensor: zeros everywhere except slack_idx
        batch_size = tf.shape(y)[0]
        correction = tf.zeros_like(y)
        idx = tf.stack(
            [
                tf.range(batch_size, dtype=tf.int32),
                tf.fill([batch_size], tf.constant(self.slack_idx, tf.int32)),
            ],
            axis=1,
        )
        correction = tf.tensor_scatter_nd_sub(
            correction, idx, tf.squeeze(diff, axis=-1)
        )
        # subtract diff from slack variable so A = L + E
        return y + correction
