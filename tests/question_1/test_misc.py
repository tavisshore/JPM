import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from jpm.question_1 import misc

unit = pytest.mark.unit
integration = pytest.mark.integration


@unit
def test_find_subtree_locates_nested_key():
    """find_subtree returns the dict value when the key exists."""
    tree = {
        "assets": {
            "current_assets": ["cash"],
            "non_current_assets": {"property": ["ppe"]},
        },
        "liabilities": {},
    }

    result = misc.find_subtree(tree, "non_current_assets")

    assert isinstance(result, dict)
    assert "property" in result


@unit
def test_collect_leaves_and_get_leaf_values():
    """collect_leaves should gather all leaves and honor optional subkeys."""
    tree = {"layer1": {"layer2": ["a", "b"], "other": ["c"]}}
    assert sorted(misc.collect_leaves(tree)) == ["a", "b", "c"]
    assert misc.get_leaf_values(tree, sub_key="layer2") == ["a", "b"]
    assert misc.get_leaf_values(tree, sub_key="missing") == []


@unit
def test_to_tensor_and_tf_sum():
    """to_tensor/tf_sum convert lists to tensors and sum elementwise."""
    tensors = [misc.to_tensor([1, 2]), misc.to_tensor([3, 4])]
    summed = misc.tf_sum(tensors)
    np.testing.assert_allclose(summed.numpy(), np.array([4.0, 6.0], dtype=np.float32))


@unit
def test_coerce_float_handles_edge_cases():
    """coerce_float must gracefully handle NaNs and invalid strings."""
    assert misc.coerce_float(np.nan) == 0.0
    assert misc.coerce_float("123.45") == pytest.approx(123.45)
    assert misc.coerce_float("bad") == 0.0


@unit
def test_as_series_fills_missing_years_with_nan():
    """as_series should insert NaNs for years that have no mapping."""
    mapping = {2020: 1.0, 2022: 3.0}
    years = [2020, 2021, 2022]
    series = misc.as_series(mapping, years)
    assert series.loc[2020] == 1.0
    assert pd.isna(series.loc[2021])
    assert series.loc[2022] == 3.0


@unit
def test_errs_below_tol_evaluates_all_entries():
    """errs_below_tol should flag when any tensor exceeds the tolerance."""
    errs = {"a": tf.constant(1e-5), "b": tf.constant(2e-4)}
    assert not bool(misc.errs_below_tol(errs, tol=1e-4).numpy())
    assert bool(misc.errs_below_tol(errs, tol=1e-3).numpy())


@unit
def test_format_money_formats_ranges():
    """format_money should emit scaled suffixes based on magnitude."""
    assert misc.format_money(500) == "$500"
    assert misc.format_money(12_000) == "$12k"
    assert misc.format_money(5_500_000) == "$5.5mn"
    assert misc.format_money(2_000_000_000) == "$2bn"
    assert misc.format_money(3_000_000_000_000) == "$3tn"
