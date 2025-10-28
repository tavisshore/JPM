"""
Valez-Pareja Full Financial Statements Model
- This is broken into 6 modules:
1. Input Data
2. Intermediate Tables
3. Cash Budget
4. Debt Schedule
5. Income Statement
6. Balance Sheet

Assumptions made with the example:
1. A startup firm (starting from zero).
2. Taxes are paid the same year as accrued
3. All the expenses and sales are paid and received on a cash basis.
4. Dividends are 100% of the Net Income of previous year and are paid the next year
after the Net Income is generated.
5. Any deficit is covered by new debt.
6. Deficit in the operating module (Module 1) should be covered with short term loans.
Short term loans will be repaid the following year.
7. Deficit in the investment in fixed assets module (Module 2) should be covered with
long term loans. Long term loans are repaid in 5 years.
8. Any cash excess above the targeted level is invested in market securities.
9. In this example we only consider two types of debt: one long term loan and short
term loans (for illustration purposes).
10. Short term portion of debt is not considered in the current liabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


def as_series(mapping, years):
    return pd.Series(
        [mapping.get(y, math.nan) for y in years], index=years, dtype="float64"
    )


@dataclass(frozen=True)
class InputData:
    years: pd.Index
    ebit: pd.Series
    depreciation: pd.Series
    net_fixed_assets: pd.Series
    min_cash: pd.Series
    kd: pd.Series  # cost of debt
    rtn_st_inv: pd.Series  # return of short-term investments
    equity_investment: float
    lt_loan_term_years: int

    def __post_init__(self):
        # Check if this is okay practice
        object.__setattr__(self, "ebit", self.ebit.fillna(0))
        object.__setattr__(self, "depreciation", self.depreciation.fillna(0))
        object.__setattr__(self, "net_fixed_assets", self.net_fixed_assets.fillna(0))
        object.__setattr__(self, "min_cash", self.min_cash.fillna(0))
        object.__setattr__(self, "kd", self.kd.fillna(0))
        object.__setattr__(self, "rtn_st_inv", self.rtn_st_inv.fillna(0))

    @property
    def ebitda(self) -> pd.Series:
        return (self.ebit + self.depreciation).fillna(0.0)


@dataclass
class BudgetState:
    cum_ncb_prev: float = 0.0  # Cumulated NCB at end of previous year
    st_invest_prev: float = 0.0  # EOY ST investments (previous year)
    st_loan_beg: float = (
        0.0  # Current year beginning ST-loan balance (= last year's draw)
    )
    lt_beg_balance: float = 0.0  # Current year beginning LT-loan balance
    lt_annual_principal: float = 0.0  # Fixed principal per year


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

    def project_cb(
        self,
        year: int,
        state: BudgetState,
        equity_contrib: float = 0.0,
        dividends: float = 0.0,
    ) -> tuple[pd.Series, BudgetState]:
        y = year
        if y not in self.I.years:
            raise ValueError(f"Year {y} not in InputData.years")

        # Module 1 - Operating Activities
        ebitda = float(self.I.ebitda.reindex(self.I.years).at[y])

        # Module 2 - Investments in assets
        capex = self._capex_for(y)
        ncb_of_invest = -capex  # Keeping temporarily to follow paper
        ncb_after_invest = -capex + ebitda

        # Module 3/4/5 - Debt, Investments, Owner transactions
        kd_t = float(
            self.I.kd.reindex(self.I.years, fill_value=0.0).at[y]
        )  # current interest rate
        # Return rate on short-term investments
        r_st_t = float(self.I.rtn_st_inv.reindex(self.I.years, fill_value=0.0).at[y])
        mincash = float(self.I.min_cash.reindex(self.I.years).at[y])

        # ST: last year repaid in full this year
        st_interest = state.st_loan_beg * kd_t
        st_principal = state.st_loan_beg
        # LT: equal-principal; if year 0 needs an initial draw, we set it below
        lt_interest = state.lt_beg_balance * kd_t
        lt_principal = min(state.lt_annual_principal, state.lt_beg_balance)
        total_debt_payment = st_interest + st_principal + lt_interest + lt_principal
        # ST Investments inflows
        st_return = state.st_invest_prev * r_st_t
        st_redeem = state.st_invest_prev
        st_inflow = st_return + st_redeem

        # LT draw only at year 0 - if initial CapEx exceeds equity, cover with LT draw
        lt_draw = 0.0
        if y == self.I.years.min():
            lt_draw = max(0.0, capex - equity_contrib)

        # Print every single variable above with name in f-string to left
        print(f"Year {y} calculations:")
        print(
            f"ebitda: {ebitda}\n"
            f", capex: {capex}\n"
            f", ncb_of_invest: {ncb_of_invest}\n"
            f", ncb_after_invest: {ncb_after_invest}\n"
            f", kd_t: {kd_t}\n"
            f", r_st_t: {r_st_t}\n"
            f", mincash: {mincash}\n"
            f", st_interest: {st_interest}\n"
            f", st_principal: {st_principal}\n"
            f", lt_interest: {lt_interest}\n"
            f", lt_principal: {lt_principal}\n"
            f", total_debt_payment: {total_debt_payment}\n"
            f", st_return: {st_return}\n"
            f", st_redeem: {st_redeem}\n"
            f", st_inflow: {st_inflow}\n"
            f", lt_draw: {lt_draw}\n"
        )

        print(
            state.cum_ncb_prev, ncb_after_invest, total_debt_payment, st_inflow, mincash
        )

        gap = (
            state.cum_ncb_prev  # cash carried from last year
            + ncb_after_invest  # operating − capex this year
            - total_debt_payment  # interest+principal paid this year
            + st_inflow  # redemptions + returns from last year's ST-investments
            - mincash  # required floor for this year
        )
        # st_loan = max(0.0, float(self.I.min_cash.at[y]) - ebitda)

        print(max(0.0, -gap))
        st_draw = 10
        if y != 0:
            st_draw = max(0.0, -gap)
        # Financing and owner transactions
        ncb_financing = st_draw + lt_draw - total_debt_payment
        ncb_owners = equity_contrib - dividends
        ncb_after_prev = ncb_owners + ncb_financing + ncb_after_invest
        # End-of-year ST investment chosen to hit MinCash
        st_invest_end = state.cum_ncb_prev + ncb_after_prev + st_inflow - mincash

        ncb_discretionary = st_inflow - st_invest_end
        ncb_for_year = ncb_discretionary + ncb_owners + ncb_financing + ncb_after_invest
        cum_ncb = state.cum_ncb_prev + ncb_for_year

        out = pd.Series(
            {
                "Operating NCB (EBITDA)": ebitda,
                "Purchase of fixed assets": capex,
                "NCB of investment in fixed assets": ncb_of_invest,
                "NCB after investment in fixed assets": ncb_after_invest,
                "ST Loan (draw)": st_draw,
                "LT Loan (draw)": lt_draw,
                "Interest ST loan": st_interest,
                "Principal ST loan": st_principal,
                "Interest LT loan": lt_interest,
                "Principal LT loan": lt_principal,
                "Total debt payment": total_debt_payment,
                "Initial invested equity": equity_contrib,
                "Dividends payment": dividends,
                "NCB of transactions with owners": ncb_owners,
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
        next_lt_beg = state.lt_beg_balance - lt_principal + lt_draw
        next_lt_ann_prin = state.lt_annual_principal
        if lt_draw > 0.0 and self.I.lt_loan_term_years > 0:
            next_lt_ann_prin = lt_draw / self.I.lt_loan_term_years

        next_state = BudgetState(
            cum_ncb_prev=cum_ncb,
            st_invest_prev=st_invest_end,
            st_loan_beg=st_draw,  # next year's beginning ST = this year's draw
            lt_beg_balance=next_lt_beg,
            lt_annual_principal=next_lt_ann_prin,
        )
        return out, next_state

    # def year0(self) -> pd.Series:
    #     """
    #     InputData and Formulas are taken from Table 5 'Cash budget for year 0'
    #     """
    #     y = 0
    #     # Module 1 - Operating Activities
    #     # Operating NCB (EBITDA) — initial assumption is 0
    #     v = pd.to_numeric(self.I.ebitda.get(y, 0.0), errors="coerce")
    #     op_ncb = 0.0 if pd.isna(v) else float(v)

    #     # Module 2 - Investment in Assets
    #     # Purchase of fixed assets = D15 (Net fixed assets, year 0)
    #     purchase_fixed_assets = float(self.I.net_fixed_assets.at[y])
    #     ncb_of_invest = -purchase_fixed_assets
    #     ncb_after_invest = ncb_of_invest + op_ncb
    #     initial_equity = float(self.I.equity_investment)

    #     # Module 3 - External Financing
    #     # ST loan (1 year) = -(Op Net Cash Bal NCB - Min cash req for year 0)
    #     st_loan = max(0.0, float(self.I.min_cash.at[y]) - op_ncb)
    #     lt_loan = max(0.0, -( ncb_of_invest + initial_equity ))
    #     total_debt_payment = 0.0 # Year 0 assumption
    #     ncb_financing = st_loan + lt_loan - total_debt_payment

    #     # Module 4 - Transactions with Owners
    #     dividends = 0.0
    #     ncb_owners = initial_equity - dividends
    #     ncb_after_prev = ncb_owners + ncb_financing + ncb_after_invest

    #     # Module 5 - Discretionary Transactions
    #     redemption_st_inv = 0.0
    #     return_st_inv = 0.0
    #     total_inflow_st_inv = 0.0
    #     st_investments = 0.0 + ncb_after_prev + total_inflow_st_inv
    #     - float(self.I.min_cash.at[y])
    #     ncb_discretionary = total_inflow_st_inv - st_investments
    #     ncb_for_year = ncb_discretionary + ncb_owners + ncb_financing
    #     + ncb_after_invest
    #     cum_ncb = ncb_for_year

    #     rows = {
    #         # Operating Activities
    #         "Operating NCB (EBITDA)": op_ncb,
    #         # Investment in Assets
    #         "Purchase of fixed assets": purchase_fixed_assets,
    #         "NCB of investment in fixed assets": ncb_of_invest,
    #         "NCB after investment in fixed assets": ncb_after_invest,
    #         # External Financing
    #         "ST Loan": st_loan,
    #         "LT Loan": lt_loan,
    #         "Total debt payment": total_debt_payment,
    #         "NCB of financing activities": ncb_financing,
    #         # Transactions with Owners
    #         "Initial invested equity": initial_equity,
    #         "Dividends payment": dividends,
    #         "NCB of transactions with owners": ncb_owners,
    #         # Discretionary Financing
    #         "Redemption of ST investment": redemption_st_inv,
    #         "Return from ST investments": return_st_inv,
    #         "Total inflow from ST investments": total_inflow_st_inv,
    #         "ST investments => BS": st_investments,
    #         "NCB of discretionary transactions": ncb_discretionary,
    #         "NCB for the year": ncb_for_year,
    #         "Cumulated NCB => BS": cum_ncb,
    #     }
    #     return pd.Series(rows, name="Year 0")


# Short-term loan: drawn in year t, fully repaid in t+1
@dataclass
class STLoanSchedule:
    years: pd.Index
    rate: pd.Series
    draws: pd.Series

    def compute(self) -> pd.DataFrame:
        y = self.years
        draws = self.draws.reindex(y, fill_value=0.0)
        rate = self.rate.reindex(y, fill_value=0.0)
        start_bal = draws.copy()

        interest = pd.Series(0.0, index=y)
        interest.iloc[1:] = (
            start_bal.shift(1).fillna(0.0).iloc[1:] * rate.iloc[1:]
        ).values
        # Principal in year t = last year start (paid off in one year)
        principal = start_bal.shift(1).fillna(0.0)
        total_payment = interest + principal
        end = pd.Series(0.0, index=y)
        end.iloc[:-1] = start_bal.iloc[:-1].values
        # at year t end, still outstanding; zero after it’s repaid next year
        # TODO - recheck if paid within the year assumption

        out = pd.DataFrame(
            {
                "Beginning balance": start_bal,
                "Interest payment ST loan": interest,
                "Principal payments ST loan": principal,
                "Total payment ST loan": total_payment,
                "Ending balance": end,
                "Interest rate": rate,
            }
        )
        return out


# Long-term loan: equal-principal amortization over `term_years`
@dataclass
class LTLoanSchedule:
    years: pd.Index
    rate: pd.Series
    initial_draw: float
    term_years: int

    def compute(self) -> pd.DataFrame:
        y = self.years

        # beginning balance trajectory (year 0 is after draw, before any payments)
        beg = pd.Series(0.0, index=y)
        beg.iloc[0] = float(self.initial_draw)

        # principal paid each payment year (equal-principal)
        annual_principal = (
            self.initial_draw / self.term_years if self.term_years > 0 else 0.0
        )

        # roll forward balances and payments
        principal = pd.Series(0.0, index=y)
        interest = pd.Series(0.0, index=y)
        end = pd.Series(0.0, index=y)

        for i, _yr in enumerate(y):
            if i == 0:
                end.iloc[i] = beg.iloc[i]  # no payment in year 0
                continue

            # beginning of year i equals end of year i-1
            beg.iloc[i] = end.iloc[i - 1]

            # interest for year i on beginning balance
            r = float(self.rate.reindex(y, fill_value=0.0).iloc[i])
            interest.iloc[i] = beg.iloc[i] * r

            # principal: equal amount until balance hits zero
            principal.iloc[i] = min(annual_principal, beg.iloc[i])
            end.iloc[i] = beg.iloc[i] - principal.iloc[i]

        total_payment = interest + principal

        out = pd.DataFrame(
            {
                "Beginning balance": beg,
                "Interest payment LT loan": interest,
                "Principal payments LT loan": principal,
                "Total payment LT loan": total_payment,
                "Ending balance": end,
                "Interest rate": self.rate.reindex(y, fill_value=0.0),
            }
        )
        return out


@dataclass
class IncomeStatement:
    years: pd.Index
    ebit: pd.Series  # InputData.ebit
    rtn_rate_st: pd.Series  # InputData.rtn_st_inv
    st_invest_end_prev: pd.Series  # EoY ST investments from CashBudget
    st_interest: pd.Series  # "Interest payment ST loan" from STLoanSchedule
    lt_interest: pd.Series  # "Interest payment LT loan" from LTLoanSchedule
    dividends: pd.Series | float = (
        0.0  # cash dividends decided this year (paid next year?)
    )

    def compute(self) -> pd.DataFrame:
        y = self.years

        # normalise inputs to the index
        ebit = self.ebit.reindex(y, fill_value=0.0)
        rtn_rate = self.rtn_rate_st.reindex(y, fill_value=0.0)
        st_inv_prev = self.st_invest_end_prev.reindex(y, fill_value=0.0)
        st_int = self.st_interest.reindex(y, fill_value=0.0)
        lt_int = self.lt_interest.reindex(y, fill_value=0.0)

        # Return from ST investments in year t is earned on prior-year invested cash
        # i.e., use previous year's ending ST-investments
        st_return = (st_inv_prev.shift(1).fillna(0.0)) * rtn_rate

        interest_total = st_int + lt_int
        net_income = ebit + st_return - interest_total

        if isinstance(self.dividends, pd.Series):
            dividends = self.dividends.reindex(y, fill_value=0.0)
        else:
            dividends = pd.Series(float(self.dividends), index=y)

        # Cumulated retained earnings (starting from 0)
        retained = (net_income - dividends).cumsum()

        df = pd.DataFrame(
            {
                "EBIT": ebit,
                "Return from ST investment": st_return,
                "Interest payments (ST+LT)": interest_total,
                "Net income": net_income,
                "Dividends (declared this year)": dividends,
                "Cumulated retained earnings": retained,
            }
        )
        return df


if __name__ == "__main__":
    years = pd.Index([0, 1, 2], name="year")
    idata = InputData(
        years=years,
        ebit=as_series({0: 0, 1: 5, 2: 9}, years),
        depreciation=as_series({0: 0, 1: 9, 2: 9}, years),
        net_fixed_assets=as_series({0: 45, 1: 36, 2: 27}, years),
        min_cash=as_series({0: 10, 1: 10, 2: 10}, years),
        kd=as_series({0: 0, 1: 0.13, 2: 0.13}, years),
        rtn_st_inv=as_series({0: 0, 1: 0.08, 2: 0.08}, years),
        equity_investment=25.0,
        lt_loan_term_years=5,
    )
    cb = CashBudget(idata)

    # Year 0
    state0 = BudgetState()
    cb0, cb_state1 = cb.project_cb(
        year=0, state=state0, equity_contrib=idata.equity_investment, dividends=0.0
    )  # Check if cb0 NaNs are better as NaN or 0.0
    # print(cb0)

    # st_loan_sched = STLoanSchedule(
    #     years=years,
    #     rate=as_series({1:0.10,2:0.10}, years),
    #     draws=as_series({0:15,1:0,2:0}, years),
    # )
    # # print(st_loan_sched.compute())
    # lt_loan_sched = LTLoanSchedule(
    #     years=years,
    #     rate=as_series({0:0.12,1:0.12,2:0.12}, years),
    #     initial_draw=20.0,
    #     term_years=5,
    # )
    # # print(lt_loan_sched.compute())

    # # CashBudget (Year 0) gave ST investments end-of-year = 0
    # st_invest_bs = pd.Series({0:0.0, 1:0.0, 2:0.0}, index=years)

    # is1 = IncomeStatement(
    #     years=years,
    #     ebit=idata.ebit,                     # {1:5, 2:9}
    #     rtn_rate_st=idata.rtn_st_inv,        # {1:0.08, 2:0.08}
    #     st_invest_end_prev=st_invest_bs,     # 0 at year 0 ⇒ return in year 1 = 0
    #     st_interest=st_loan_sched.compute()["Interest payment ST loan"],
    #     lt_interest=lt_loan_sched.compute()["Interest payment LT loan"],
    #     dividends=0.0,
    # ).compute()

    # # Subsequent years - loop with some example data

    # print(is1.loc[1].round(1))
    print()
