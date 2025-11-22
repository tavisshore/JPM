import pytest

from jpm.question_1.config import Config
from jpm.question_1.models.income_statement import IncomeStatement
from jpm.question_1.models.metrics import Metric, TickerResults

unit = pytest.mark.unit


@unit
def test_income_statement_view_shows_baseline_comparison(capsys):
    """IncomeStatement.view should print baseline comparison with GT/pred/error."""
    features = {
        "revenue_from_contract_with_customer_excluding_assessed_tax": Metric(
            value=200.0, mae=0.0, gt=190.0
        ),
        "cost_of_goods_and_services_sold": Metric(value=100.0, mae=0.0, gt=90.0),
    }

    results = TickerResults(
        assets=Metric(0, 0),
        liabilities=Metric(0, 0),
        equity=Metric(0, 0),
        features=features,
        net_income_gt=100.0,
        net_income_pred=110.0,
        net_income_baseline_pred={"baseline": 95.0},
        net_income_baseline_mae={"baseline": 10.0},
        net_income_skill={"baseline": -0.1},
    )

    stmt = IncomeStatement(Config(), results)
    stmt.view()
    out = capsys.readouterr().out

    assert "Income Statement" in out
    assert "Baseline Comparison (Net Income)" in out
    # Strip ANSI codes and check key values are present
    import re

    plain = re.sub(r"\x1b\[[0-9;]*m", "", out)
    assert "$100.0" in plain  # ground truth
    assert "$100.0" in plain  # model pred (derived from sections)
    assert "$95.0" in plain  # baseline pred
