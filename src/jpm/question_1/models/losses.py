import tensorflow as tf
from tensorflow.keras.layers import Layer

from jpm.question_1.config import LossConfig


def bs_loss(
    feature_means,
    feature_stds,
    feature_mappings,
    config: LossConfig,
):
    """Build a balance-sheet-aware loss combining MSE with identity penalties."""
    means64 = tf.constant(feature_means, dtype=tf.float64)
    stds64 = tf.constant(feature_stds, dtype=tf.float64)

    asset_idx_tf = tf.constant(feature_mappings["assets"], dtype=tf.int32)
    liability_idx_tf = tf.constant(feature_mappings["liabilities"], dtype=tf.int32)
    equity_idx_tf = tf.constant(feature_mappings["equity"], dtype=tf.int32)

    if config.learn_subtotals:
        asset_cur_idx_tf = tf.constant(
            feature_mappings["current_assets"], dtype=tf.int32
        )
        asset_noncur_idx_tf = tf.constant(
            feature_mappings["non_current_assets"], dtype=tf.int32
        )
        liab_cur_idx_tf = tf.constant(
            feature_mappings["current_liabilities"], dtype=tf.int32
        )
        liab_noncur_idx_tf = tf.constant(
            feature_mappings["non_current_liabilities"], dtype=tf.int32
        )

    def loss(y_true, y_pred):
        """Compute loss for a batch."""
        y_true64 = tf.cast(y_true, tf.float64)
        y_pred64 = tf.cast(y_pred, tf.float64)

        base = tf.reduce_mean(tf.square(y_true64 - y_pred64))

        y_pred_unscaled = y_pred64 * stds64 + means64

        assets = tf.reduce_sum(
            tf.gather(y_pred_unscaled, asset_idx_tf, axis=-1), axis=-1
        )
        liabilities = tf.reduce_sum(
            tf.gather(y_pred_unscaled, liability_idx_tf, axis=-1), axis=-1
        )
        equity = tf.reduce_sum(
            tf.gather(y_pred_unscaled, equity_idx_tf, axis=-1), axis=-1
        )

        eps = tf.constant(1e-6, dtype=tf.float64)

        total_loss = base

        if config.learn_identity:
            # Enforce Assets ≈ Liabilities + Equity on unscaled values
            violation = assets - (liabilities + equity)
            denom_ALE = tf.abs(assets) + tf.abs(liabilities) + tf.abs(equity) + eps
            rel_violation = violation / denom_ALE
            identity_penalty = tf.reduce_mean(tf.square(rel_violation))
            total_loss = total_loss + config.identity_weight * identity_penalty

        if config.learn_subtotals:
            assets_current = tf.reduce_sum(
                tf.gather(y_pred_unscaled, asset_cur_idx_tf, axis=-1), axis=-1
            )
            assets_noncurrent = tf.reduce_sum(
                tf.gather(y_pred_unscaled, asset_noncur_idx_tf, axis=-1), axis=-1
            )
            assets_sub_sum = assets_current + assets_noncurrent
            assets_sub_violation = assets_sub_sum - assets
            denom_A = tf.abs(assets) + eps
            rel_assets_sub_violation = assets_sub_violation / denom_A
            assets_sub_penalty = tf.reduce_mean(tf.square(rel_assets_sub_violation))

            liab_current = tf.reduce_sum(
                tf.gather(y_pred_unscaled, liab_cur_idx_tf, axis=-1), axis=-1
            )
            liab_noncurrent = tf.reduce_sum(
                tf.gather(y_pred_unscaled, liab_noncur_idx_tf, axis=-1), axis=-1
            )
            liab_sub_sum = liab_current + liab_noncurrent
            liab_sub_violation = liab_sub_sum - liabilities
            denom_L = tf.abs(liabilities) + eps
            rel_liab_sub_violation = liab_sub_violation / denom_L
            liab_sub_penalty = tf.reduce_mean(tf.square(rel_liab_sub_violation))

            subcategory_penalty = assets_sub_penalty + liab_sub_penalty
            # Subtotal consistency penalties are down-weighted
            total_loss = total_loss + config.subcategory_weight * subcategory_penalty

        return total_loss

    return loss


class EnforceBalance(Layer):
    """Enforce A = L + E via a slack equity term"""

    def __init__(
        self,
        feature_mappings,
        feature_means,
        feature_stds,
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

        if (
            not feature_mappings["assets"]
            or not feature_mappings["liabilities"]
            or not feature_mappings["equity"]
        ):
            raise ValueError(
                "feature_mappings for assets, liabilities, and equity must be non-empty"
            )

        if feature_names is None:
            raise ValueError("feature_names required to find slack index")

        if slack_name not in feature_names:
            raise ValueError(
                f"Slack variable '{slack_name}' not found in feature names"
            )

        self.slack_idx = feature_names[slack_name]

        # Use float64 to minimise drift when rescaling and adjusting slack
        self.means = tf.constant(feature_means, dtype=tf.float64)
        self.stds = tf.constant(feature_stds, dtype=tf.float64)

    def call(self, y):
        y64 = tf.cast(y, tf.float64)

        y_unscaled = y64 * self.stds + self.means

        assets = tf.reduce_sum(
            tf.gather(y_unscaled, self.asset_idx, axis=-1), axis=-1, keepdims=True
        )
        liabilities = tf.reduce_sum(
            tf.gather(y_unscaled, self.liability_idx, axis=-1), axis=-1, keepdims=True
        )
        equity = tf.reduce_sum(
            tf.gather(y_unscaled, self.equity_idx, axis=-1), axis=-1, keepdims=True
        )

        diff = assets - (liabilities + equity)

        batch_size = tf.shape(y_unscaled)[0]
        idx = tf.stack(
            [
                tf.range(batch_size, dtype=tf.int32),
                tf.fill([batch_size], tf.constant(self.slack_idx, tf.int32)),
            ],
            axis=1,
        )

        y_unscaled_corrected = tf.tensor_scatter_nd_add(
            y_unscaled, idx, tf.squeeze(diff, axis=-1)
        )

        # Rescale back to network space, then cast to y dtype
        y_scaled64 = (y_unscaled_corrected - self.means) / self.stds
        return tf.cast(y_scaled64, y.dtype)
