import types
from unittest.mock import Mock

import pytest

import jpm.question_1.models.balance_sheet as bs_mod
from jpm.question_1.models.balance_sheet import (
    Assets,
    BalanceSheet,
    Equity,
    Liabilities,
)

unit = pytest.mark.unit
integration = pytest.mark.integration


class DummyConfig:
    class Data:
        def __init__(self, ticker: str) -> None:
            self.ticker = ticker

    def __init__(self, ticker: str) -> None:
        self.data = DummyConfig.Data(ticker=ticker)


class DummyResults:
    def __init__(self, feature_values: dict) -> None:
        self._feature_values = feature_values

    def feature_values(self) -> dict:
        return self._feature_values


@pytest.fixture
def feature_values():
    return {
        # Current assets
        "cash_and_cash_equivalents_at_carrying_value": 100.0,
        "marketable_securities_current": 200.0,
        "accounts_receivable_net_current": 300.0,
        "nontrade_receivables_current": 50.0,
        "inventory_net": 400.0,
        "other_assets_current": 25.0,
        # Non-current assets
        "marketable_securities_noncurrent": 600.0,
        "other_assets_noncurrent": 80.0,
        "property_plant_and_equipment_net": 900.0,
        # Current liabilities
        "accounts_payable_current": 150.0,
        "contract_with_customer_liability_current": 70.0,
        "other_liabilities_current": 30.0,
        "commercial_paper": 40.0,
        "other_short_term_borrowings": 60.0,
        "long_term_debt_current": 55.0,
        # Non-current liabilities
        "other_liabilities_noncurrent": 500.0,
        "long_term_debt_noncurrent": 700.0,
        # Equity
        "retained_earnings_accumulated_deficit": 1200.0,
        "common_stocks_including_additional_paid_in_capital": 1500.0,
        "accumulated_other_comprehensive_income_loss_net_of_tax": 100.0,
        # Unused field to verify ignored data doesn't break anything
        "unused_metric": 999.0,
    }


@pytest.fixture
def bs_structure():
    return {
        "assets": {
            "current_assets": [
                "cash_and_cash_equivalents_at_carrying_value",
                "marketable_securities_current",
                "accounts_receivable_net_current",
                "nontrade_receivables_current",
                "inventory_net",
                "other_assets_current",
            ],
            "non_current_assets": [
                "marketable_securities_noncurrent",
                "other_assets_noncurrent",
                "property_plant_and_equipment_net",
            ],
        },
        "liabilities": {
            "current_liabilities": [
                "accounts_payable_current",
                "contract_with_customer_liability_current",
                "other_liabilities_current",
                "commercial_paper",
                "other_short_term_borrowings",
                "long_term_debt_current",
            ],
            "non_current_liabilities": [
                "other_liabilities_noncurrent",
                "long_term_debt_noncurrent",
            ],
        },
        "equity": [
            "retained_earnings_accumulated_deficit",
            "common_stocks_including_additional_paid_in_capital",
            "accumulated_other_comprehensive_income_loss_net_of_tax",
        ],
    }


@pytest.fixture
def config():
    return DummyConfig(ticker="TEST")


@pytest.fixture
def balance_sheet(config, feature_values, bs_structure, monkeypatch):
    # Patch get_bs_structure to return our controlled structure
    monkeypatch.setattr(
        bs_mod,
        "get_bs_structure",
        lambda ticker: bs_structure,
    )

    results = DummyResults(feature_values)
    return BalanceSheet(config=config, results=results)


@integration
def test_balance_sheet_initialization_fetches_structure_and_features(
    config, feature_values, bs_structure, monkeypatch
):
    """Ensure BalanceSheet grabs feature values and ticker structure exactly once."""
    results = Mock()
    results.feature_values = Mock(return_value=feature_values)

    get_bs_structure_mock = Mock(return_value=bs_structure)
    monkeypatch.setattr(bs_mod, "get_bs_structure", get_bs_structure_mock)

    bs = BalanceSheet(config=config, results=results)

    results.feature_values.assert_called_once_with()
    get_bs_structure_mock.assert_called_once_with(ticker=config.data.ticker)
    assert bs._feature_values is feature_values
    assert bs.bs_structure is bs_structure


# --- Dataclass property tests -------------------------------------------------


@unit
def test_assets_total():
    """Assets.total should sum current and non-current buckets."""
    assets = Assets(
        current_assets={"a": 10.0, "b": 5.0},
        non_current_assets={"c": 20.0},
    )
    assert assets.total == 35.0


@unit
def test_liabilities_total():
    """Liabilities.total should sum both current and non-current portions."""
    liab = Liabilities(
        current_liabilities={"a": 7.0},
        non_current_liabilities={"b": 3.0, "c": 10.0},
    )
    assert liab.total == 20.0


@unit
def test_equity_total():
    """Equity.total simply sums all tracked equity items."""
    eq = Equity(
        items={"cs": 100.0, "re": 50.0},
    )
    assert eq.total == 150.0


# --- Building from structure + feature_values ---------------------------------


@integration
def test_build_assets_uses_structure_and_feature_values(balance_sheet):
    """Builder should read the configured metric names and fetch the matching values."""
    assets = balance_sheet.assets

    # Current assets
    assert assets.current_assets[
        "cash_and_cash_equivalents_at_carrying_value"
    ] == pytest.approx(100.0)
    assert assets.current_assets["inventory_net"] == pytest.approx(400.0)

    # Non-current assets
    assert assets.non_current_assets[
        "property_plant_and_equipment_net"
    ] == pytest.approx(900.0)
    assert assets.non_current_assets["other_assets_noncurrent"] == pytest.approx(80.0)


@integration
def test_build_liabilities_uses_structure_and_feature_values(balance_sheet):
    """Liabilities builder should fetch everything listed for both buckets."""
    liab = balance_sheet.liabilities

    # Current liabilities
    assert liab.current_liabilities["accounts_payable_current"] == pytest.approx(150.0)
    assert liab.current_liabilities["long_term_debt_current"] == pytest.approx(55.0)

    # Non-current liabilities
    assert liab.non_current_liabilities["long_term_debt_noncurrent"] == pytest.approx(
        700.0
    )
    assert liab.non_current_liabilities[
        "other_liabilities_noncurrent"
    ] == pytest.approx(500.0)


@integration
def test_build_equity_uses_structure_and_feature_values(balance_sheet):
    """Equity builder loads all configured equity metrics via feature values."""
    eq = balance_sheet.equity

    assert eq.items[
        "common_stocks_including_additional_paid_in_capital"
    ] == pytest.approx(1500.0)
    assert eq.items["retained_earnings_accumulated_deficit"] == pytest.approx(1200.0)
    assert eq.items[
        "accumulated_other_comprehensive_income_loss_net_of_tax"
    ] == pytest.approx(100.0)


@integration
def test_build_sections_handle_missing_structure_and_default_zero(config, monkeypatch):
    """Sections missing names should default to empty dicts and zero-filled values."""
    custom_structure = {
        "assets": {
            "current_assets": ["known_current_asset", "missing_current_asset"],
            # non_current_assets intentionally omitted to ensure default empty dict
        },
        "liabilities": {
            # current_liabilities intentionally omitted to ensure default empty dict
            "non_current_liabilities": [
                "known_non_current_liab",
                "missing_non_current",
            ],
        },
        "equity": ["known_equity", "missing_equity"],
    }
    feature_values = {
        "known_current_asset": 75.5,
        "known_non_current_liab": 210.0,
        "known_equity": 333.3,
    }

    monkeypatch.setattr(bs_mod, "get_bs_structure", lambda ticker: custom_structure)
    results = DummyResults(feature_values)
    bs = BalanceSheet(config=config, results=results)

    assert bs.assets.current_assets == {
        "known_current_asset": pytest.approx(75.5),
        "missing_current_asset": pytest.approx(0.0),
    }
    assert bs.assets.non_current_assets == {}

    assert bs.liabilities.current_liabilities == {}
    assert bs.liabilities.non_current_liabilities == {
        "known_non_current_liab": pytest.approx(210.0),
        "missing_non_current": pytest.approx(0.0),
    }

    assert bs.equity.items == {
        "known_equity": pytest.approx(333.3),
        "missing_equity": pytest.approx(0.0),
    }


@integration
def test_builders_ignore_unreferenced_metrics(balance_sheet):
    """Data outside the structure should never bleed into the balance-sheet sections."""
    assert "unused_metric" not in balance_sheet.assets.current_assets
    assert "unused_metric" not in balance_sheet.assets.non_current_assets
    assert "unused_metric" not in balance_sheet.liabilities.current_liabilities
    assert "unused_metric" not in balance_sheet.liabilities.non_current_liabilities
    assert "unused_metric" not in balance_sheet.equity.items


@unit
def test_get_value_missing_key_returns_zero(config, bs_structure, monkeypatch):
    """Any missing metric should be interpreted as 0.0 rather than raising errors."""
    feature_values = {"existing": 42.0}
    results = DummyResults(feature_values)

    monkeypatch.setattr(
        bs_mod,
        "get_bs_structure",
        lambda ticker: bs_structure,
    )

    bs = BalanceSheet(config=config, results=results)

    assert bs._get_value("nonexistent_key") == 0.0


@unit
def test_get_value_converts_to_float(balance_sheet):
    """_get_value should coerce ints/strings into floats for downstream math."""
    balance_sheet._feature_values["int_metric"] = 7
    balance_sheet._feature_values["string_metric"] = "8.5"

    assert balance_sheet._get_value("int_metric") == pytest.approx(7.0)
    assert balance_sheet._get_value("string_metric") == pytest.approx(8.5)


@integration
def test_total_properties_match_component_totals(balance_sheet):
    """Top-level BalanceSheet totals should simply mirror the underlying objects."""
    assert balance_sheet.total_assets == pytest.approx(2655.0)
    assert balance_sheet.total_liabilities == pytest.approx(1605.0)
    assert balance_sheet.total_equity == pytest.approx(2800.0)
    assert balance_sheet.total_liabilities_and_equity == pytest.approx(4405.0)


@integration
def test_total_liabilities_and_equity_uses_component_totals(
    empty_structure, monkeypatch
):
    """Regression test to ensure liabilities plus equity equals
    the sum of both components."""
    bs = _make_bs_with_totals(
        assets_total=0.0,
        liabilities_total=111.0,
        equity_total=222.0,
        empty_structure=empty_structure,
        monkeypatch=monkeypatch,
    )

    assert bs.total_liabilities_and_equity == pytest.approx(333.0)


# --- check_identity behaviour + colour selection ------------------------------


@pytest.fixture
def empty_structure():
    # For identity tests we override totals directly, so we don't need real names
    return {
        "assets": {
            "current_assets": [],
            "non_current_assets": [],
        },
        "liabilities": {
            "current_liabilities": [],
            "non_current_liabilities": [],
        },
        "equity": [],
    }


def _make_bs_with_totals(
    assets_total, liabilities_total, equity_total, empty_structure, monkeypatch
):
    config = DummyConfig(ticker="IDENTITY")
    results = DummyResults(feature_values={})

    monkeypatch.setattr(
        bs_mod,
        "get_bs_structure",
        lambda ticker: empty_structure,
    )

    # Typing isn't ideal here
    bs = BalanceSheet(config=config, results=results)

    # Overwrite with simple objects that just expose a 'total' attribute
    bs.assets = types.SimpleNamespace(total=float(assets_total))
    bs.liabilities = types.SimpleNamespace(total=float(liabilities_total))
    bs.equity = types.SimpleNamespace(total=float(equity_total))

    return bs


@integration
def test_check_identity_zero_assets_handles_division_by_zero(
    empty_structure, monkeypatch
):
    """Zero assets should not explode diff_pct and still report cleanly."""
    bs = _make_bs_with_totals(
        assets_total=0.0,
        liabilities_total=0.0,
        equity_total=150.0,
        empty_structure=empty_structure,
        monkeypatch=monkeypatch,
    )

    colour_mock = Mock(side_effect=lambda text, c: f"COLOUR({text},{c})")
    format_money_mock = Mock(side_effect=lambda x: f"FMT({x})")
    print_table_mock = Mock()

    monkeypatch.setattr(bs_mod, "colour", colour_mock)
    monkeypatch.setattr(bs_mod, "format_money", format_money_mock)
    monkeypatch.setattr(bs_mod, "print_table", print_table_mock)

    bs.check_identity(atol=1.0)  # diff= -150 -> fail, but pct should be 0.0%
