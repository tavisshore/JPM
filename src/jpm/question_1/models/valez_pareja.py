from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

# Constructing Consistent Financial Planning Models for Valuation
# Separates FS into 3 models: Income Statement (IS) ->
# 2. Balance Sheet (BS) -> 3. Cash Flow Statement (CFS)
# We will use these models to project future financials and perform
# valuation analysis
# Models must be constructed sequentially to both maintain all
# constraints and predict future cash flows accurately


@dataclass
class IncomeStatement:
    """
    Represents an Income Statement (P&L Statement).

    Attributes:
        revenue: Total sales or operating revenue.
        cogs: Cost of goods sold.
        operating_expenses: Selling, general, and administrative expenses.
        depreciation: Non-cash depreciation and amortization expense.
        interest_expense: Interest paid on debt.
        tax_rate: Effective tax rate (as a decimal, e.g., 0.25 for 25%).
    """

    # --- Core Line Items ---
    revenue: Optional[float] = field(default=None)
    cogs: Optional[float] = field(default=None)
    operating_expenses: Optional[float] = field(default=None)
    depreciation: Optional[float] = field(default=None)
    interest_expense: Optional[float] = field(default=None)
    tax_rate: Optional[float] = field(default=None)

    # --- Computed Values (to be implemented later) ---
    def gross_profit(self) -> Optional[float]:
        """Compute Gross Profit = Revenue - COGS"""
        # TODO: Implement formula
        pass

    def operating_income(self) -> Optional[float]:
        """Operating Income = Gross Profit -
        Operating Expenses - Depreciation"""
        # TODO: Implement formula
        pass

    def pretax_income(self) -> Optional[float]:
        """Compute Pretax Income = Operating Income - Interest Expense"""
        # TODO: Implement formula
        pass

    def net_income(self) -> Optional[float]:
        """Compute Net Income = Pretax Income * (1 - Tax Rate)"""
        # TODO: Implement formula
        pass

    # --- Utility Methods ---
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IncomeStatement":
        """
        Create an IncomeStatement instance from a dictionary.
        Extra keys in `data` are ignored.
        """
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the IncomeStatement instance to a dictionary."""
        return asdict(self)
