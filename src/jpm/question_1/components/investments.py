from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.jpm.question_1.components.input import InputData


@dataclass
class STInvestment:
    """ """


@dataclass
class InvestmentBook:
    """
    Bookkeeping for investments
    - Short-term investments
    """

    input: InputData
    st_investments: List[STInvestment] = field(default_factory=list)

    def add(self, investment: STInvestment) -> None:
        self.st_investments.append(investment)

    def total_st_investment_at_end(self, year: object) -> float:
        total = 0.0
        for inv in self.st_investments:
            if year in inv._df.index:
                total += inv._df.at[year, "end_balance"]
        return total
