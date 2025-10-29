from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.jpm.question_1.components.input import InputData
from src.jpm.question_1.components.investments import Investment, InvestmentBook
from src.jpm.question_1.components.loans import Loan, LoanBook


@dataclass
class CashBudget:
    """
    Table 5 for Multi-Year - independent calculation, no circularity
    """

    input: InputData
    years: pd.Index
    cum_ncb: pd.Series = field(init=False, repr=False)

    def __post_init__(self):
        z = pd.Series(0.0, index=self.years)
        object.__setattr__(self, "cum_ncb", z.copy())

    def _capex_for(self, year: int) -> float:
        nfa = self.input.net_fixed_assets
        dep = self.input.depreciation
        if year == self.input.years.min():
            return float(nfa.at[year])
        prev = self.input.years[self.input.years.get_loc(year) - 1]
        return float(nfa.at[year] - nfa.at[prev] + dep.at[year])

    def new_st_loan(
        self, prev_cum_ncb, ncb_after_invest, total_debt_payment, st_inflow, min_cash
    ):
        x = prev_cum_ncb + ncb_after_invest - total_debt_payment + st_inflow - min_cash
        return 0 if x > 0 else -x

    def new_st_investment(self, prev_cum_ncb, min_cash):
        x = prev_cum_ncb - min_cash
        return x if x > 0 else 0

    def project_cb(
        self,
        year: int,
        loanbook: LoanBook,
        investmentbook: InvestmentBook,
        equity_contrib: float = 0.0,
        dividends: float = 0.0,
    ) -> pd.Series:
        y = year
        if y not in self.input.years:
            raise ValueError(f"Year {y} not in InputData.years")

        mincash = float(self.input.min_cash.reindex(self.input.years).at[y])
        loan_record = loanbook.debt_payments(year=year)
        investment_record = investmentbook.investment_income(year=year)

        # Module 1 - Operating Activities
        ebitda = float(self.input.ebitda.reindex(self.input.years).at[y])

        # Module 2 - Investments in assets
        purchase_fixed_assets = self._capex_for(y)  # Purchase of fixed assets
        ncb_of_invest = -purchase_fixed_assets  # NCB of investment in fixed assets
        ncb_after_invest = ncb_of_invest + ebitda

        # Module 3 - External Financing
        # Deficit in Module 2 -> Take out LT Loan
        # So if ncb_after_invest < 0, cover with LT loan draw ?
        if ncb_after_invest < 0.0:
            lt_loan_draw = -ncb_after_invest
            loanbook.add(
                Loan(
                    input=self.input,
                    start_year=y,
                    initial_draw=lt_loan_draw,
                    category="LT",
                )
            )
        else:
            lt_loan_draw = 0.0

        # + Module 5 - Discretionary transactions
        # ST: last year repaid in full this year
        st_loan_draw = self.new_st_loan(
            self.cum_ncb.loc[year - 1],
            ncb_after_invest,
            loan_record.total,
            investment_record.total,
            mincash,
        )

        # TODO - check functionality on later years 2+
        if st_loan_draw:
            loanbook.add(
                Loan(
                    input=self.input,
                    initial_draw=st_loan_draw,
                    start_year=y,
                    category="ST",
                )
            )

        # + Module 4 - Transactions with Owners
        ncb_financing = st_loan_draw + lt_loan_draw - loan_record.total
        ncb_owners = equity_contrib - dividends
        ncb_after_prev = ncb_owners + ncb_financing + ncb_after_invest
        # End-of-year ST investment chosen to hit MinCash
        st_invest_end = (
            self.cum_ncb.loc[year - 1]
            + ncb_after_prev
            + investment_record.total
            - mincash
        )
        if st_invest_end:
            investmentbook.add(
                Investment(
                    input=self.input,
                    amount=st_invest_end,
                    start_year=y,
                    term_years=1,
                )
            )

        ncb_discretionary = investment_record.total - st_invest_end
        ncb_for_year = ncb_discretionary + ncb_owners + ncb_financing + ncb_after_invest
        cum_ncb = self.cum_ncb.loc[year - 1] + ncb_for_year

        out = pd.Series(
            {
                # Operating Activities
                "Operating NCB (EBITDA)": ebitda,
                # Investment in Assets
                "Purchase of fixed assets": purchase_fixed_assets,
                "NCB of investment in fixed assets": ncb_of_invest,
                "NCB after investment in fixed assets": ncb_after_invest,
                # External Financing
                "ST Loan": st_loan_draw,
                "LT Loan": lt_loan_draw,
                "Principal ST loan": loan_record.st_principal,  # all of these?
                "Interest ST loan": loan_record.st_interest,
                "Principal LT loan": loan_record.lt_principal,
                "Interest LT loan": loan_record.lt_interest,
                "Total debt payment": loan_record.total,  # To here
                "NCB of financing activities": ncb_financing,
                # Transactions with Owners
                "Initial invested equity": equity_contrib,
                "Dividends payment": dividends,
                "NCB of transactions with owners": ncb_owners,
                "NCB for the year after previous transactions": ncb_after_prev,
                # Discretionary Financing
                "Redemption of ST investment": investment_record.st_principal_in,
                "Return from ST investments": investment_record.st_interest,
                "Total inflow from ST investments": investment_record.total,
                "ST investments => BS": st_invest_end,  # To here
                "NCB of discretionary transactions": ncb_discretionary,
                "NCB for the year": ncb_for_year,
                "Cumulated NCB => BS": cum_ncb,
            },
            name=y,
        )

        self.cum_ncb.loc[year] = cum_ncb
        return out

    def year0(self, loanbook: LoanBook, investmentbook: InvestmentBook) -> pd.Series:
        """
        InputData and Formulas are taken from Table 5 'Cash budget for year 0'
        - Keeping separate for now - assuming real data won't go from this init
        """
        y = 0
        # Module 1 - Operating Activities
        # Operating NCB (EBITDA) — initial assumption is 0
        ebitda = float(
            pd.to_numeric(self.input.ebitda.get(y, 0.0), errors="coerce") or 0.0
        )

        # Module 2 - Investment in Assets
        # Purchase of fixed assets = D15 (Net fixed assets, year 0)
        purchase_fixed_assets = float(self.input.net_fixed_assets.at[y])
        ncb_of_invest = -purchase_fixed_assets
        ncb_after_invest = ncb_of_invest + ebitda
        initial_equity = float(self.input.equity_investment)

        # Module 3 - External Financing
        # ST loan (1 year) = -(Op Net Cash Bal NCB - Min cash req for year 0)
        st_loan = max(0.0, float(self.input.min_cash.at[y]) - ebitda)
        lt_loan = max(0.0, -(ncb_of_invest + initial_equity))

        # Store instances of loans for later use
        if st_loan:
            loanbook.add(
                Loan(
                    input=self.input, start_year=y, initial_draw=st_loan, category="ST"
                )
            )
        if lt_loan:
            loanbook.add(
                Loan(
                    input=self.input, start_year=y, initial_draw=lt_loan, category="LT"
                )
            )

        total_debt_payment = 0.0  # Year 0 assumption
        ncb_financing = st_loan + lt_loan - total_debt_payment

        # Module 4 - Transactions with Owners
        dividends = 0.0
        ncb_owners = initial_equity - dividends
        ncb_after_prev = ncb_owners + ncb_financing + ncb_after_invest

        # Module 5 - Discretionary Transactions
        redemption_st_inv = 0.0
        return_st_inv = 0.0
        total_inflow_st_inv = 0.0
        st_investments = (
            0.0
            + ncb_after_prev
            + total_inflow_st_inv
            - float(self.input.min_cash.at[y])
        )
        # Create new investment
        if st_investments:
            investmentbook.add(
                Investment(
                    input=self.input,
                    amount=st_investments,
                    start_year=y,
                    term_years=1,
                )
            )

        ncb_discretionary = total_inflow_st_inv - st_investments
        ncb_for_year = ncb_discretionary + ncb_owners + ncb_financing + ncb_after_invest
        cum_ncb = ncb_for_year

        rows = {
            # Operating Activities
            "Operating NCB (EBITDA)": ebitda,
            # Investment in Assets
            "Purchase of fixed assets": purchase_fixed_assets,
            "NCB of investment in fixed assets": ncb_of_invest,
            "NCB after investment in fixed assets": ncb_after_invest,
            # External Financing
            "ST Loan": st_loan,
            "LT Loan": lt_loan,
            "Total debt payment": total_debt_payment,
            "NCB of financing activities": ncb_financing,
            # Transactions with Owners
            "Initial invested equity": initial_equity,
            "Dividends payment": dividends,
            "NCB of transactions with owners": ncb_owners,
            "NCB for the year after previous transactions": ncb_after_prev,
            # Discretionary Financing
            "Redemption of ST investment": redemption_st_inv,
            "Return from ST investments": return_st_inv,
            "Total inflow from ST investments": total_inflow_st_inv,
            "ST investments => BS": st_investments,
            "NCB of discretionary transactions": ncb_discretionary,
            "NCB for the year": ncb_for_year,
            "Cumulated NCB => BS": cum_ncb,
        }
        self.cum_ncb.loc[y] = cum_ncb
        return pd.Series(rows, name="Year 0")
