from dataclasses import dataclass

import pandas as pd
import tensorflow as tf

from .cash import CashBudget
from .income_statement import IncomeStatement
from .input import InputData
from .trans import Transactions


@dataclass
class CashFlow:
    # CFD
    loan_inflows: float
    pp: float
    ip: float
    cfd: float
    ncb_module_3: float

    # CFE
    ei: float
    div: float
    sr: float
    cfe: float
    ncb_module_4: float

    # CCF and FCF
    ccf: float
    ts: float
    fcf: float

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
        cfd = float(loan_inflows + pp + ip)
        ncb_module_3 = cash_budget.ncb_financing_activities

        # CFE
        ei = -transactions.owner.invested_equity.iloc[-1]
        div = transactions.owner.dividends.iloc[-1]
        sr = transactions.owner.repurchased_stock.iloc[-1]
        cfe = (ei + div + sr).numpy().item()
        ncb_module_4 = float(transactions.owner.ncb_with_owners.iloc[-1])

        # CCF and FCF
        ccf = cfd + cfe
        base = (
            tf.add(income_statement.ebit, income_statement.return_from_st_investment)
            .numpy()
            .item()
        )
        interest_payments = income_statement.interest_payments

        clipped = max(min(base, interest_payments), 0)

        ts = input_data.corporate_tax_rate * clipped

        fcf = ccf - ts

        return cls(
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

    def pretty_print(self) -> None:
        """Pretty print CashFlow data."""
        lines = ["=" * 60, "Cash Flow Statement", "=" * 60]

        lines.append("\nCASH FLOW FROM DEBT (CFD):")
        lines.append("-" * 60)
        lines.append("Loan Inflows:")
        lines.append(str(self.loan_inflows))
        lines.append("\nPrincipal Payments:")
        lines.append(str(self.pp))
        lines.append("\nInterest Payments:")
        lines.append(str(self.ip))
        lines.append("\nCFD (Total):")
        lines.append(str(self.cfd))
        lines.append("\nNCB Module 3:")
        lines.append(str(self.ncb_module_3))

        lines.append("\n" + "=" * 60)
        lines.append("CASH FLOW FROM EQUITY (CFE):")
        lines.append("-" * 60)
        lines.append("Equity Investment:")
        lines.append(str(self.ei))
        lines.append("\nDividends:")
        lines.append(str(self.div))
        lines.append("\nStock Repurchase:")
        lines.append(str(self.sr))
        lines.append("\nCFE (Total):")
        lines.append(str(self.cfe))
        lines.append("\nNCB Module 4:")
        lines.append(str(self.ncb_module_4))

        lines.append("\n" + "=" * 60)
        lines.append("COMBINED CASH FLOW:")
        lines.append("-" * 60)
        lines.append("CCF (CFD + CFE):")
        lines.append(str(self.ccf))
        lines.append("\nTax Shield:")
        lines.append(str(self.ts))
        lines.append("\nFCF (Free Cash Flow):")
        lines.append(str(self.fcf))

        lines.append("=" * 60)

        print("\n".join(lines))
