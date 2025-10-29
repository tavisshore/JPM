from __future__ import annotations

import pandas as pd

from src.jpm.question_1.components.input import BudgetState, InputData
from src.jpm.question_1.components.loans import LoanBook, LTLoan, STLoan


class CashBudget:
    """
    Table 5 for Multi-Year - independent calculation, no circularity
    """

    def __init__(self, inputs: InputData):
        self.I = inputs
        if 0 not in self.I.years:
            raise ValueError("Year 0 must be present in InputData.years")

    def _capex_for(self, year: int) -> float:
        nfa = self.I.net_fixed_assets
        dep = self.I.depreciation
        if year == self.I.years.min():
            return float(nfa.at[year])
        prev = self.I.years[self.I.years.get_loc(year) - 1]
        return float(nfa.at[year] - nfa.at[prev] + dep.at[year])

    def new_st_loan(
        self, prev_cum_ncb, ncb_after_invest, total_debt_payment, st_inflow, min_cash
    ):
        x = prev_cum_ncb + ncb_after_invest - total_debt_payment + st_inflow - min_cash
        return 0 if x > 0 else -x

    def project_cb(
        self,
        year: int,
        state: BudgetState,
        loanbook: LoanBook,
        equity_contrib: float = 0.0,
        dividends: float = 0.0,
    ) -> tuple[pd.Series, BudgetState]:
        y = year
        if y not in self.I.years:
            raise ValueError(f"Year {y} not in InputData.years")

        mincash = float(self.I.min_cash.reindex(self.I.years).at[y])

        # Module 1 - Operating Activities
        ebitda = float(self.I.ebitda.reindex(self.I.years).at[y])

        # Module 2 - Investments in assets
        purchase_fixed_assets = self._capex_for(y)  # Purchase of fixed assets
        ncb_of_invest = -purchase_fixed_assets  # NCB of investment in fixed assets
        ncb_after_invest = ncb_of_invest + ebitda

        # Module 3 - External Financing
        loan_record = loanbook.debt_payments(year=year)

        # LT draw only at year 0 - if initial CapEx exceeds equity, cover with LT draw
        # NOTE Add functionality for new long-term loans later
        lt_draw = 0.0

        # + Module 5 - Discretionary transactions
        st_return_rate = float(
            self.I.rtn_st_inv.reindex(self.I.years, fill_value=0.0).at[y]
        )
        st_return = state.st_invest_prev * st_return_rate
        st_redeem = state.st_invest_prev
        st_inflow = st_return + st_redeem

        # ST: last year repaid in full this year
        st_loan = self.new_st_loan(
            state.cum_ncb_prev, ncb_after_invest, loan_record.total, st_inflow, mincash
        )

        # Add new loans
        # TODO - add loan functionality, getting dues etc.
        if st_loan:  # Better way - unit test for negative events
            loanbook.add(STLoan(input=self.I, amount=st_loan, start_year=y))
        if lt_draw:
            loanbook.add(LTLoan(input=self.I, start_year=y, initial_draw=lt_draw))

        # + Module 4 - Transactions with Owners
        ncb_financing = st_loan + lt_draw - loan_record.total
        ncb_owners = equity_contrib - dividends
        ncb_after_prev = ncb_owners + ncb_financing + ncb_after_invest
        # End-of-year ST investment chosen to hit MinCash
        st_invest_end = state.cum_ncb_prev + ncb_after_prev + st_inflow - mincash

        ncb_discretionary = st_inflow - st_invest_end
        ncb_for_year = ncb_discretionary + ncb_owners + ncb_financing + ncb_after_invest
        cum_ncb = state.cum_ncb_prev + ncb_for_year

        out = pd.Series(
            {
                # Operating Activities
                "Operating NCB (EBITDA)": ebitda,
                # Investment in Assets
                "Purchase of fixed assets": purchase_fixed_assets,
                "NCB of investment in fixed assets": ncb_of_invest,
                "NCB after investment in fixed assets": ncb_after_invest,
                # External Financing
                "ST Loan": st_loan,
                "LT Loan": lt_draw,
                "Interest ST loan": loan_record.st_interest,  # all of these?
                "Principal ST loan": loan_record.st_principal,
                "Interest LT loan": loan_record.lt_interest,
                "Principal LT loan": loan_record.lt_principal,
                "Total debt payment": loan_record.total,  # To here
                "NCB of financing activities": ncb_financing,
                # Transactions with Owners
                "Initial invested equity": equity_contrib,
                "Dividends payment": dividends,
                "NCB of transactions with owners": ncb_owners,
                # Discretionary Financing
                "Redemption of ST investment": st_redeem,
                "Return from ST investments": st_return,
                "Total inflow from ST investments": st_inflow,
                "ST investments => BS": st_invest_end,
                "NCB of discretionary transactions": ncb_discretionary,
                "NCB for the year": ncb_for_year,
                "Cumulated NCB => BS": cum_ncb,
            },
            name=y,
        )

        # State for the following year
        next_lt_beg = state.lt_beg_balance - loan_record.lt_principal + lt_draw
        next_lt_ann_prin = state.lt_annual_principal
        if lt_draw > 0.0 and self.I.lt_loan_term_years > 0:
            next_lt_ann_prin = lt_draw / self.I.lt_loan_term_years

        next_state = BudgetState(
            cum_ncb_prev=cum_ncb,
            st_invest_prev=st_invest_end,
            st_loan_beg=st_loan,  # next year's beginning ST = this year's draw
            lt_beg_balance=next_lt_beg,
            lt_annual_principal=next_lt_ann_prin,
        )
        return out, next_state

    def year0(self, loanbook: LoanBook) -> pd.Series:
        """
        InputData and Formulas are taken from Table 5 'Cash budget for year 0'
        - Keeping separate for now - assuming real data won't go from this init
        """
        y = 0
        # Module 1 - Operating Activities
        # Operating NCB (EBITDA) — initial assumption is 0
        ebitda = float(pd.to_numeric(self.I.ebitda.get(y, 0.0), errors="coerce") or 0.0)

        # Module 2 - Investment in Assets
        # Purchase of fixed assets = D15 (Net fixed assets, year 0)
        purchase_fixed_assets = float(self.I.net_fixed_assets.at[y])
        ncb_of_invest = -purchase_fixed_assets
        ncb_after_invest = ncb_of_invest + ebitda
        initial_equity = float(self.I.equity_investment)

        # Module 3 - External Financing
        # ST loan (1 year) = -(Op Net Cash Bal NCB - Min cash req for year 0)
        st_loan = max(0.0, float(self.I.min_cash.at[y]) - ebitda)
        lt_loan = max(0.0, -(ncb_of_invest + initial_equity))

        # Store instances of loans for later use
        if st_loan:
            loanbook.add(STLoan(input=self.I, amount=st_loan, start_year=y))
        if lt_loan:
            loanbook.add(LTLoan(input=self.I, start_year=y, initial_draw=lt_loan))

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
            0.0 + ncb_after_prev + total_inflow_st_inv - float(self.I.min_cash.at[y])
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
            # Discretionary Financing
            "Redemption of ST investment": redemption_st_inv,
            "Return from ST investments": return_st_inv,
            "Total inflow from ST investments": total_inflow_st_inv,
            "ST investments => BS": st_investments,
            "NCB of discretionary transactions": ncb_discretionary,
            "NCB for the year": ncb_for_year,
            "Cumulated NCB => BS": cum_ncb,
        }
        return pd.Series(rows, name="Year 0")
