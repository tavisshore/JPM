"""
Simple model based on the work:
Constructing Consistent Financial Planning Models for Valuation
"""

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional

from src.jpm.question_1.data.utils import title_to_snake


# Outlined in 9
@dataclass
class Assumptions:
    plugs: bool = field(default=False)
    circularity: bool = field(default=False)
    new_firm: bool = field(default=True)
    same_year_tax: bool = field(default=True)


class InputData:
    # Equity
    ordinary_shares_number: Optional[float] = field(default=None)
    share_issued: Optional[float] = field(default=None)
    common_stock_equity: Optional[float] = field(default=None)
    stockholders_equity: Optional[float] = field(default=None)
    total_equity_gross_minority_interest: Optional[float] = field(default=None)
    capital_stock: Optional[float] = field(default=None)
    common_stock: Optional[float] = field(default=None)
    retained_earnings: Optional[float] = field(default=None)
    gains_losses_not_affecting_retained_earnings: Optional[float] = field(default=None)
    other_equity_adjustments: Optional[float] = field(default=None)

    # Assets

    # Liabilities

    @classmethod
    def from_dicts(cls, data: Dict[str, Dict]) -> "InputData":
        field_names = {f.name for f in fields(cls)}
        init_kwargs = {}
        for _key_1, dict_1 in data.items():
            for key, value in dict_1.items():
                snake_key = title_to_snake(key)
                if snake_key in field_names:
                    init_kwargs[snake_key] = value
        return cls(**init_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
