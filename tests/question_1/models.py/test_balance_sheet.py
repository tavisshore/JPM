import pytest

from jpm.question_1.models.validation.balance_sheet import (
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


class DummyDataset:
    """Mock dataset with bs_structure."""

    def __init__(self, bs_structure: dict) -> None:
        self.bs_structure = bs_structure


class DummyData:
    """Mock EdgarData."""

    pass


@pytest.fixture
def feature_values():
    return {
        # Assets
        "Cash and Equivalents": 100.0,
        "Receivables": 200.0,
        "Inventory": 300.0,
        "Property, Plant, and Equipment (net)": 400.0,
        "Intangible Assets (net)": 150.0,
        # Liabilities
        "Accounts Payable and Accrued Expenses": 150.0,
        "Short-term Debt": 70.0,
        "Long-term Debt": 500.0,
        "Other Non-Current Liabilities": 80.0,
        # Equity
        "Common Stock and APIC": 1200.0,
        "Retained Earnings": 800.0,
        "Accumulated Other Comprehensive Income": 50.0,
        # Unused field to verify ignored data doesn't break anything
        "unused_metric": 999.0,
    }


@pytest.fixture
def bs_structure():
    return {
        "Assets": [
            "Cash and Equivalents",
            "Receivables",
            "Inventory",
            "Property, Plant, and Equipment (net)",
            "Intangible Assets (net)",
        ],
        "Liabilities": [
            "Accounts Payable and Accrued Expenses",
            "Short-term Debt",
            "Long-term Debt",
            "Other Non-Current Liabilities",
        ],
        "Equity": [
            "Common Stock and APIC",
            "Retained Earnings",
            "Accumulated Other Comprehensive Income",
        ],
    }


@pytest.fixture
def config():
    return DummyConfig(ticker="TEST")


@pytest.fixture
def balance_sheet(config, feature_values, bs_structure):
    results = DummyResults(feature_values)
    dataset = DummyDataset(bs_structure)
    data = DummyData()
    return BalanceSheet(config=config, data=data, dataset=dataset, results=results)


# --- Dataclass property tests -------------------------------------------------


@unit
def test_assets_total():
    """Assets.total should sum all asset items."""
    assets = Assets(
        assets={"a": 10.0, "b": 5.0, "c": 20.0},
    )
    assert assets.total == 35.0


@unit
def test_liabilities_total():
    """Liabilities.total should sum all liability items."""
    liab = Liabilities(
        liabilities={"a": 7.0, "b": 3.0, "c": 10.0},
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

    assert assets.assets["Cash and Equivalents"] == pytest.approx(100.0)
    assert assets.assets["Inventory"] == pytest.approx(300.0)
    assert assets.assets["Property, Plant, and Equipment (net)"] == pytest.approx(400.0)


@integration
def test_build_liabilities_uses_structure_and_feature_values(balance_sheet):
    """Liabilities builder should fetch everything listed in the structure."""
    liab = balance_sheet.liabilities

    assert liab.liabilities["Accounts Payable and Accrued Expenses"] == pytest.approx(
        150.0
    )
    assert liab.liabilities["Long-term Debt"] == pytest.approx(500.0)


@integration
def test_build_equity_uses_structure_and_feature_values(balance_sheet):
    """Equity builder loads all configured equity metrics via feature values."""
    eq = balance_sheet.equity

    assert eq.items["Common Stock and APIC"] == pytest.approx(1200.0)
    assert eq.items["Retained Earnings"] == pytest.approx(800.0)
    assert eq.items["Accumulated Other Comprehensive Income"] == pytest.approx(50.0)


@integration
def test_builders_ignore_unreferenced_metrics(balance_sheet):
    """Data outside the structure should never bleed into the balance-sheet sections."""
    assert "unused_metric" not in balance_sheet.assets.assets
    assert "unused_metric" not in balance_sheet.liabilities.liabilities
    assert "unused_metric" not in balance_sheet.equity.items


@unit
def test_get_value_missing_key_returns_zero(config, bs_structure):
    """Any missing metric should be interpreted as 0.0 rather than raising errors."""
    feature_values = {"existing": 42.0}
    results = DummyResults(feature_values)
    dataset = DummyDataset(bs_structure)
    data = DummyData()

    bs = BalanceSheet(config=config, data=data, dataset=dataset, results=results)

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
    # Assets: 100 + 200 + 300 + 400 + 150 = 1150
    assert balance_sheet.total_assets == pytest.approx(1150.0)
    # Liabilities: 150 + 70 + 500 + 80 = 800
    assert balance_sheet.total_liabilities == pytest.approx(800.0)
    # Equity: 1200 + 800 + 50 = 2050
    assert balance_sheet.total_equity == pytest.approx(2050.0)
    # L + E = 800 + 2050 = 2850
    assert balance_sheet.total_liabilities_and_equity == pytest.approx(2850.0)


@integration
def test_check_identity_returns_diff_pct(balance_sheet):
    """check_identity should return difference percentage."""
    diff_pct = balance_sheet.check_identity(atol=1e6, verbose=False)
    # A=1150, L+E=2850 -> diff = 1150 - 2850 = -1700
    # diff_pct = -1700 / 1150 * 100 ≈ -147.83%
    expected_diff_pct = ((1150 - 2850) / 1150) * 100
    assert diff_pct == pytest.approx(expected_diff_pct)


@unit
def test_check_identity_zero_assets_handles_division_by_zero(config):
    """Zero assets should not explode diff_pct and still report cleanly."""
    results = DummyResults({})
    bs_structure = {"Assets": [], "Liabilities": [], "Equity": []}
    dataset = DummyDataset(bs_structure)
    data = DummyData()

    bs = BalanceSheet(config=config, data=data, dataset=dataset, results=results)

    # Manually add equity to test zero-division handling
    bs.equity = Equity(items={"equity_item": 150.0})

    # Zero assets means diff_pct should be 0.0 (not inf or nan)
    diff_pct = bs.check_identity(atol=1.0, verbose=False)
    assert diff_pct == 0.0
