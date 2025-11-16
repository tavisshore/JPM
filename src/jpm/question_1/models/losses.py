import tensorflow as tf
from tensorflow.keras.layers import Layer

from src.jpm.question_1.config import LossConfig


def bs_loss(
    feature_means,
    feature_stds,
    feature_mappings,
    config: LossConfig,
):
    """
    Balance-sheet loss:
      - base MSE in scaled space
      - A = L + E penalty in *relative* terms in unscaled space
      - (optional) current/non-current subcategory consistency in *relative* terms:
          (assets_current + assets_noncurrent) ≈ assets_total
          (liabilities_current + liabilities_noncurrent) ≈ liabilities_total
    """
    means_tf = tf.constant(feature_means, dtype=tf.float32)
    stds_tf = tf.constant(feature_stds, dtype=tf.float32)

    if config.learn_identity:
        # A, L, E index tensors
        asset_idx_tf = tf.constant(feature_mappings["assets"], dtype=tf.int32)
        liability_idx_tf = tf.constant(feature_mappings["liabilities"], dtype=tf.int32)
        equity_idx_tf = tf.constant(feature_mappings["equity"], dtype=tf.int32)

    # Optional subtotal index tensors
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
        # base loss = MSE in scaled space
        total_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        if config.learn_identity:
            # Unscale predictions
            y_pred_unscaled = y_pred * stds_tf + means_tf  # (batch, F)

            # ---- A, L, E sums in unscaled space ----
            assets = tf.reduce_sum(
                tf.gather(y_pred_unscaled, asset_idx_tf, axis=-1), axis=-1
            )  # (batch,)
            liabilities = tf.reduce_sum(
                tf.gather(y_pred_unscaled, liability_idx_tf, axis=-1), axis=-1
            )
            equity = tf.reduce_sum(
                tf.gather(y_pred_unscaled, equity_idx_tf, axis=-1), axis=-1
            )

            # ---- Relative A = L + E penalty ----
            violation = assets - (liabilities + equity)
            eps = tf.constant(1e-6, dtype=tf.float32)
            denom_ALE = tf.abs(assets) + tf.abs(liabilities) + tf.abs(equity) + eps
            rel_violation = violation / denom_ALE
            identity_penalty = tf.reduce_mean(tf.square(rel_violation))

            total_loss = total_loss + config.identity_weight * identity_penalty

        # ---- Optional current/non-current consistency (also relative) ----
        if config.learn_subtotals:
            # Assets: current + non-current ≈ total
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

            # Liabilities: current + non-current ≈ total
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
            total_loss = total_loss + config.subcategory_weight * subcategory_penalty

        return total_loss

    return loss


class EnforceBalance(Layer):
    """ """

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

        if feature_names is None:
            raise ValueError("feature_names required to find slack index")

        if slack_name not in feature_names:
            raise ValueError(
                f"Slack variable '{slack_name}' not found in feature names"
            )

        self.slack_idx = feature_names.index(slack_name)

        equity_idx_py = list(feature_mappings["equity"])
        if self.slack_idx not in equity_idx_py:
            raise ValueError(
                f"Slack index {self.slack_idx} ('{slack_name}') "
                f"is not in equity indices {equity_idx_py}. "
                f"Include the slack feature in the equity mapping."
            )

        self.means = tf.constant(feature_means, dtype=tf.float32)
        self.stds = tf.constant(feature_stds, dtype=tf.float32)

    def call(self, y):
        y_unscaled = y * self.stds + self.means  # (batch, F)

        # Compute unscaled A, L, E
        assets = tf.reduce_sum(
            tf.gather(y_unscaled, self.asset_idx, axis=-1), axis=-1, keepdims=True
        )
        liabilities = tf.reduce_sum(
            tf.gather(y_unscaled, self.liability_idx, axis=-1), axis=-1, keepdims=True
        )
        equity = tf.reduce_sum(
            tf.gather(y_unscaled, self.equity_idx, axis=-1), axis=-1, keepdims=True
        )

        diff = assets - (liabilities + equity)  # (batch, 1)

        # Subtract diff from slack - unscaled
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

        # Rescaling to network space
        return (y_unscaled_corrected - self.means) / self.stds
