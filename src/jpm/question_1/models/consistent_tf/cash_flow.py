from dataclasses import dataclass

import pandas as pd

from .cash import CashBudget
from .income_statement import IncomeStatement
from .input import InputData
from .trans import Transactions


@dataclass
class CashFlow:
    years: pd.Index

    # CFD
    loan_inflows: pd.Series
    pp: pd.Series
    ip: pd.Series
    cfd: pd.Series
    ncb_module_3: pd.Series

    # CFE
    ei: pd.Series
    div: pd.Series
    sr: pd.Series
    cfe: pd.Series
    ncb_module_4: pd.Series

    # CCF and FCF
    ccf: pd.Series
    ts: pd.Series
    fcf: pd.Series

    @classmethod
    def from_inputs(
        cls,
        years: pd.Index,
        input_data: "InputData",
        cash_budget: "CashBudget",
        transactions: "Transactions",
        income_statement: "IncomeStatement",
    ) -> "CashFlow":
        years = pd.Index(years)

        # CFD
        loan_inflows = -(cash_budget.st_loan_inflow + cash_budget.lt_loan_inflow)
        pp = cash_budget.principal_st_loan + cash_budget.principal_lt_loan
        ip = cash_budget.interest_st_loan + cash_budget.interest_lt_loan
        cfd = loan_inflows + pp + ip
        ncb_module_3 = cash_budget.ncb_financing_activities

        # CFE
        ei = -transactions.owner.invested_equity
        div = transactions.owner.dividends
        sr = transactions.owner.repurchased_stock
        cfe = ei + div + sr
        ncb_module_4 = transactions.owner.ncb_with_owners

        # CCF and FCF
        ccf = cfd + cfe
        base = income_statement.ebit + income_statement.return_from_st_investment

        ts = input_data.corporate_tax_rate * (
            base.clip(upper=income_statement.interest_payments).clip(
                lower=0
            )  # element-wise min  # element-wise max
        )

        fcf = ccf - ts

        return cls(
            years=years,
            loan_inflows=loan_inflows,
            pp=pp,
            ip=ip,
            cfd=cfd,
            ncb_module_3=ncb_module_3,
            ei=ei,
            div=div,
            sr=sr,
            cfe=cfe,
            ncb_module_4=ncb_module_4,
            ccf=ccf,
            ts=ts,
            fcf=fcf,
        )
