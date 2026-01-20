import numpy as np
import pytest
import tensorflow as tf

from jpm.question_1.config import LSTMConfig
from jpm.question_1.models.losses import EnforceBalance, bs_loss

unit = pytest.mark.unit
integration = pytest.mark.integration


@unit
def test_bs_loss_returns_mse_when_regularizers_disabled():
    """With all constraints disabled, bs_loss should reduce to vanilla MSE."""
    feature_means = [0.0, 0.0, 0.0, 0.0]
    feature_stds = [1.0, 1.0, 1.0, 1.0]
    feature_mappings = {
        "assets": [0, 1],
        "liabilities": [2],
        "equity": [3],
    }
    config = LSTMConfig(
        learn_identity=False,
        learn_subtotals=False,
    )

    loss_fn = bs_loss(feature_means, feature_stds, feature_mappings, config)

    y_true = tf.constant([[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]], dtype=tf.float32)
    y_pred = tf.constant(
        [[0.9, 2.1, 3.5, 3.0], [0.1, 1.2, 1.2, 2.5]],
        dtype=tf.float32,
    )

    expected = tf.reduce_mean(
        tf.square(tf.cast(y_true, tf.float64) - tf.cast(y_pred, tf.float64))
    ).numpy()

    assert loss_fn(y_true, y_pred).numpy() == pytest.approx(expected)


@unit
def test_bs_loss_includes_identity_penalty_when_enabled():
    """Identity penalty should be added to the base loss when configured."""
    feature_means = [0.0, 0.0, 0.0, 0.0]
    feature_stds = [1.0, 1.0, 1.0, 1.0]
    feature_mappings = {
        "assets": [0, 1],
        "liabilities": [2],
        "equity": [3],
    }
    config = LSTMConfig(
        learn_identity=True,
        identity_weight=2.0,
        learn_subtotals=False,
    )

    loss_fn = bs_loss(feature_means, feature_stds, feature_mappings, config)

    y_true = tf.constant(
        [[1.0, 1.0, 1.0, 1.0], [0.2, 0.1, 0.3, 0.4]],
        dtype=tf.float32,
    )
    y_pred = tf.constant(
        [[1.0, 0.5, 0.2, 0.1], [0.3, 0.4, 0.1, 0.0]],
        dtype=tf.float32,
    )

    base_loss = tf.reduce_mean(
        tf.square(tf.cast(y_true, tf.float64) - tf.cast(y_pred, tf.float64))
    ).numpy()

    y_pred_unscaled = y_pred.numpy()
    assets = np.sum(y_pred_unscaled[:, feature_mappings["assets"]], axis=1)
    liabilities = np.sum(y_pred_unscaled[:, feature_mappings["liabilities"]], axis=1)
    equity = np.sum(y_pred_unscaled[:, feature_mappings["equity"]], axis=1)
    eps = 1e-6
    denom = np.abs(assets) + np.abs(liabilities) + np.abs(equity) + eps
    rel_violation = (assets - (liabilities + equity)) / denom
    identity_penalty = np.mean(rel_violation**2)

    expected = base_loss + config.identity_weight * identity_penalty

    assert loss_fn(y_true, y_pred).numpy() == pytest.approx(expected, rel=1e-6)


@integration
def test_bs_loss_includes_subtotal_penalties_when_enabled():
    """Subtotal penalties should be applied to both assets and liabilities."""
    feature_means = [0.0] * 6
    feature_stds = [1.0] * 6
    feature_mappings = {
        "assets": [0, 1, 2],
        "liabilities": [3, 4],
        "equity": [5],
        "current_assets": [0],
        "non_current_assets": [1],  # omit index 2 to induce violation
        "current_liabilities": [3],
        "non_current_liabilities": [],  # omit index 4 to induce violation
    }
    config = LSTMConfig(
        learn_identity=False,
        learn_subtotals=True,
        subcategory_weight=3.5,
    )

    loss_fn = bs_loss(feature_means, feature_stds, feature_mappings, config)

    y_true = tf.constant(
        [[2.0, 3.0, 5.0, 7.0, 11.0, 13.0], [1.0, 1.0, 2.0, 1.0, 3.0, 5.0]],
        dtype=tf.float32,
    )
    y_pred = tf.constant(
        [[2.5, 2.0, 4.0, 6.0, 12.0, 13.0], [0.5, 1.5, 1.5, 0.5, 2.5, 4.5]],
        dtype=tf.float32,
    )

    base_loss = tf.reduce_mean(
        tf.square(tf.cast(y_true, tf.float64) - tf.cast(y_pred, tf.float64))
    ).numpy()

    y_pred_unscaled = y_pred.numpy()

    assets = np.sum(y_pred_unscaled[:, feature_mappings["assets"]], axis=1)
    assets_current = np.sum(
        y_pred_unscaled[:, feature_mappings["current_assets"]], axis=1
    )
    if feature_mappings["non_current_assets"]:
        assets_noncurrent = np.sum(
            y_pred_unscaled[:, feature_mappings["non_current_assets"]], axis=1
        )
    else:
        assets_noncurrent = np.zeros_like(assets_current)
    assets_sub_sum = assets_current + assets_noncurrent
    denom_A = np.abs(assets) + 1e-6
    assets_penalty = np.mean(((assets_sub_sum - assets) / denom_A) ** 2)

    liabilities = np.sum(y_pred_unscaled[:, feature_mappings["liabilities"]], axis=1)
    liab_current = np.sum(
        y_pred_unscaled[:, feature_mappings["current_liabilities"]], axis=1
    )
    if feature_mappings["non_current_liabilities"]:
        liab_noncurrent = np.sum(
            y_pred_unscaled[:, feature_mappings["non_current_liabilities"]], axis=1
        )
    else:
        liab_noncurrent = np.zeros_like(liab_current)
    liab_sub_sum = liab_current + liab_noncurrent
    denom_L = np.abs(liabilities) + 1e-6
    liab_penalty = np.mean(((liab_sub_sum - liabilities) / denom_L) ** 2)

    subtotal_penalty = assets_penalty + liab_penalty
    expected = base_loss + config.subcategory_weight * subtotal_penalty

    assert loss_fn(y_true, y_pred).numpy() == pytest.approx(expected, rel=1e-6)


@integration
def test_bs_loss_combines_identity_and_subtotals_with_scaling():
    """When both toggles are on, bs_loss should honour scaling and weights together."""
    feature_means = [10.0, 5.0, 2.0, -1.0, 0.5]
    feature_stds = [2.0, 0.5, 1.5, 3.0, 0.25]
    feature_mappings = {
        "assets": [0, 1],
        "liabilities": [2, 3],
        "equity": [4],
        "current_assets": [0],
        "non_current_assets": [1],
        "current_liabilities": [2],
        "non_current_liabilities": [3],
    }
    config = LSTMConfig(
        learn_identity=True,
        identity_weight=0.75,
        learn_subtotals=True,
        subcategory_weight=0.4,
    )

    loss_fn = bs_loss(feature_means, feature_stds, feature_mappings, config)

    y_true = tf.constant(
        [[0.0, 0.1, -0.2, 0.3, 0.4], [1.0, -0.5, 0.25, -0.75, 0.8]],
        dtype=tf.float32,
    )
    y_pred = tf.constant(
        [[1.0, -1.0, 0.5, 0.0, -0.5], [0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=tf.float32,
    )

    y_true64 = tf.cast(y_true, tf.float64).numpy()
    y_pred64 = tf.cast(y_pred, tf.float64).numpy()
    base_loss = np.mean((y_true64 - y_pred64) ** 2)

    # Unscale predictions
    y_pred_unscaled = y_pred64 * feature_stds + feature_means

    assets = np.sum(y_pred_unscaled[:, feature_mappings["assets"]], axis=1)
    liabilities = np.sum(y_pred_unscaled[:, feature_mappings["liabilities"]], axis=1)
    equity = np.sum(y_pred_unscaled[:, feature_mappings["equity"]], axis=1)
    eps = 1e-6
    denom_identity = np.abs(assets) + np.abs(liabilities) + np.abs(equity) + eps
    rel_violation = (assets - (liabilities + equity)) / denom_identity
    identity_penalty = np.mean(rel_violation**2)

    assets_current = np.sum(
        y_pred_unscaled[:, feature_mappings["current_assets"]], axis=1
    )
    assets_noncurrent = np.sum(
        y_pred_unscaled[:, feature_mappings["non_current_assets"]], axis=1
    )
    assets_sub_sum = assets_current + assets_noncurrent
    denom_A = np.abs(assets) + eps
    assets_penalty = np.mean(((assets_sub_sum - assets) / denom_A) ** 2)

    liab_current = np.sum(
        y_pred_unscaled[:, feature_mappings["current_liabilities"]], axis=1
    )
    liab_noncurrent = np.sum(
        y_pred_unscaled[:, feature_mappings["non_current_liabilities"]], axis=1
    )
    liab_sub_sum = liab_current + liab_noncurrent
    denom_L = np.abs(liabilities) + eps
    liab_penalty = np.mean(((liab_sub_sum - liabilities) / denom_L) ** 2)
    subtotal_penalty = assets_penalty + liab_penalty

    expected = (
        base_loss
        + config.identity_weight * identity_penalty
        + config.subcategory_weight * subtotal_penalty
    )

    assert loss_fn(y_true, y_pred).numpy() == pytest.approx(expected, rel=1e-6)


@unit
def test_enforce_balance_requires_feature_names():
    """Feature names are mandatory so slack index can be resolved."""
    feature_mappings = {"assets": [0], "liabilities": [1], "equity": [2]}

    with pytest.raises(ValueError, match="feature_names"):
        EnforceBalance(
            feature_mappings=feature_mappings,
            feature_means=[0.0, 0.0, 0.0],
            feature_stds=[1.0, 1.0, 1.0],
            feature_names=None,
        )


@unit
def test_enforce_balance_raises_if_slack_missing_from_names(monkeypatch):
    """Slack variable must exist in the supplied feature_names."""
    from jpm.question_1.models import losses

    # Mock get_slack_name to return a name not in feature_names
    monkeypatch.setattr(losses, "get_slack_name", lambda: "slack_equity")

    feature_mappings = {"assets": [0], "liabilities": [1], "equity": [2]}

    with pytest.raises(ValueError, match="Slack variable"):
        EnforceBalance(
            feature_mappings=feature_mappings,
            feature_means=[0.0, 0.0, 0.0],
            feature_stds=[1.0, 1.0, 1.0],
            feature_names={"asset": 0, "liability": 1, "equity": 2},
        )


@unit
def test_enforce_balance_raises_if_slack_not_in_feature_names(monkeypatch):
    """Slack variable must exist in feature_names dict."""
    from jpm.question_1.models import losses

    # Mock get_slack_name to return a specific name
    monkeypatch.setattr(losses, "get_slack_name", lambda: "missing_slack")

    feature_mappings = {
        "assets": [0],
        "liabilities": [1],
        "equity": [2],
    }

    with pytest.raises(ValueError, match="Slack variable"):
        EnforceBalance(
            feature_mappings=feature_mappings,
            feature_means=[0.0, 0.0, 0.0],
            feature_stds=[1.0, 1.0, 1.0],
            feature_names={"asset": 0, "liability": 1, "equity": 2},
        )


@integration
def test_enforce_balance_adjusts_only_slack_to_restore_identity(monkeypatch):
    """The layer should adjust the slack equity feature so that A = L + E holds."""
    from jpm.question_1.models import losses

    # Mock get_slack_name where it's used (in losses module)
    monkeypatch.setattr(losses, "get_slack_name", lambda: "slack_equity")

    feature_names = {
        "asset_current": 0,
        "asset_noncurrent": 1,
        "liability_current": 2,
        "liability_noncurrent": 3,
        "slack_equity": 4,
    }
    feature_mappings = {
        "assets": [0, 1],
        "liabilities": [2, 3],
        "equity": [4],
    }

    layer = EnforceBalance(
        feature_mappings=feature_mappings,
        feature_means=[0.0] * 5,
        feature_stds=[1.0] * 5,
        feature_names=feature_names,
    )

    inputs = tf.constant(
        [
            [10.0, 5.0, 8.0, 4.0, 1.0],  # diff = 10+5 - (8+4+1) = 2
            [7.5, 2.5, 4.0, 3.0, -1.0],  # diff = 7.5+2.5 - (4+3-1) = 4.0
        ],
        dtype=tf.float32,
    )

    outputs = layer(inputs)
    outputs_np = outputs.numpy()

    # Only the slack index should change
    np.testing.assert_allclose(
        outputs_np[:, :4], inputs.numpy()[:, :4], rtol=0, atol=1e-6
    )

    # Check that the corrected unscaled values now satisfy the identity
    assets = np.sum(outputs_np[:, :2], axis=1)
    liabilities = np.sum(outputs_np[:, 2:4], axis=1)
    equity = outputs_np[:, 4]
    assert np.allclose(assets, liabilities + equity, atol=1e-6)
    assert outputs.dtype == inputs.dtype


@integration
def test_enforce_balance_handles_non_unit_means_and_stds(monkeypatch):
    """Slack correction should happen in unscaled space and respect arbitrary stats."""
    from jpm.question_1.models import losses

    # Mock get_slack_name where it's used (in losses module)
    monkeypatch.setattr(losses, "get_slack_name", lambda: "slack_equity")

    feature_names = {"asset_a": 0, "asset_b": 1, "liability": 2, "slack_equity": 3}
    feature_mappings = {
        "assets": [0, 1],
        "liabilities": [2],
        "equity": [3],
    }
    feature_means = [100.0, 50.0, 80.0, 20.0]
    feature_stds = [10.0, 5.0, 8.0, 2.0]

    layer = EnforceBalance(
        feature_mappings=feature_mappings,
        feature_means=feature_means,
        feature_stds=feature_stds,
        feature_names=feature_names,
    )

    inputs = tf.constant([[0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
    outputs = layer(inputs)

    outputs_np = outputs.numpy()

    # Compute unscaled values to verify the correction
    unscaled = outputs_np * feature_stds + feature_means
    assets = np.sum(unscaled[:, feature_mappings["assets"]], axis=1)
    liabilities = np.sum(unscaled[:, feature_mappings["liabilities"]], axis=1)
    equity = unscaled[:, feature_mappings["equity"]]

    assert np.allclose(assets, liabilities + equity, atol=1e-6)
