from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CashBudget:
    years: pd.Index

    # -------- Table 9a: Cash Budget – Module 1: Operating activities --------
    # Row 125 – Inflows from sales (= D116 from Table 8b: total inflows)
    inflows_from_sales: pd.Series

    # Row 126 – Total inflows (here just equals inflows_from_sales)
    total_operating_inflows: pd.Series

    # Row 128 – Payments for purchases (= D120 from Table 8b: total outflows)
    payments_for_purchases: pd.Series

    # Row 129 – Administrative and selling expenses (= D100)
    admin_selling_expenses: pd.Series

    # Row 130 – Income taxes (= D205)
    income_taxes: pd.Series

    # Row 131 – Total cash outflows = payments_for_purchases + admin + taxes
    total_operating_outflows: pd.Series

    # Row 132 – Operating NCB = total_operating_inflows - total_operating_outflows
    operating_ncb: pd.Series

    # -------- Table 9b: Cash Budget – Module 2: Investing activities --------
    # Row 134 – Investment in fixed assets (= D76)
    investment_in_fixed_assets: pd.Series

    # Row 135 – NCB of investment in assets = -investment_in_fixed_assets
    ncb_investment_assets: pd.Series

    # Row 136 – NCB after Capex = ncb_investment_assets + operating_ncb
    ncb_after_capex: pd.Series

    # -------- Table 9c: Module 3 – External financing (filled later) --------
    # Row 139 – ST loan inflow
    st_loan_inflow: pd.Series

    # Row 140 – LT loan inflow
    lt_loan_inflow: pd.Series

    # Row 142 – Principal ST loan
    principal_st_loan: pd.Series

    # Row 143 – Interest ST loan
    interest_st_loan: pd.Series

    # Row 144 – Total ST loan payment = principal_st_loan + interest_st_loan
    total_st_loan_payment: pd.Series

    # Row 145 – Principal LT loan
    principal_lt_loan: pd.Series

    # Row 146 – Interest LT loan
    interest_lt_loan: pd.Series

    # Row 147 – Total loan payment = total_st_loan_payment +
    # principal_lt_loan + interest_lt_loan
    total_loan_payment: pd.Series

    # Row 148 – NCB of financing activities = st_loan_inflow +
    # lt_loan_inflow - total_loan_payment
    ncb_financing_activities: pd.Series

    # --------------------------------------------------------------------- #
    # Constructors
    # --------------------------------------------------------------------- #
    @classmethod
    def initial(
        cls,
        policy,
        sales_purch_schedule,
        admin_selling_expenses,
        dep_and_investment,
        forecast,
        transactions,
    ):
        years_full = sales_purch_schedule.years
        y0 = years_full[0]  # single year
        years = pd.Index([y0])  # restrict class to year 0 only

        # --- Extract ONLY year 0 values ---
        inflow0 = sales_purch_schedule.total_inflows.loc[y0]
        outflow0 = sales_purch_schedule.total_outflows.loc[y0]
        admin0 = admin_selling_expenses.total_as_expenses.loc[y0]

        inflows_from_sales = pd.Series([inflow0], index=years)
        total_operating_inflows = inflows_from_sales.copy()
        payments_for_purchases = pd.Series([outflow0], index=years)
        admin_selling_expenses = pd.Series([admin0], index=years)
        total_operating_outflows = pd.Series([outflow0 + admin0], index=years)
        operating_ncb = pd.Series([inflow0 - (outflow0 + admin0)], index=years)

        # --- Module 9b (investing) ---
        investment_in_fixed_assets = dep_and_investment.new_fixed_assets.loc[years]
        ncb_investment_assets = -investment_in_fixed_assets
        ncb_after_capex = operating_ncb + ncb_investment_assets
        income_taxes = pd.Series([0.0], index=years)

        # Loans - setting payments for year 0 to zero
        st_loan_pp = pd.Series([0.0], index=years)
        st_loan_ip = pd.Series([0.0], index=years)
        st_loan_total = pd.Series([0.0], index=years)
        lt_loan_pp = pd.Series([0.0], index=years)
        lt_loan_ip = pd.Series([0.0], index=years)
        total_loan_payment = pd.Series([0.0], index=years)

        # Now st and lt
        st_loan = 0.0
        if 0 + operating_ncb - 0 - forecast.minimum_cash_required.loc[0] > 0:
            st_loan = -(0 + operating_ncb - 0 - forecast.minimum_cash_required.loc[0])

        payments_to_owners = transactions.owner_transactions.payments_to_owners.loc[y0]
        lt_loan = 0.0
        in_from_st_investment = 0.0

        if (
            0
            + ncb_after_capex
            + st_loan
            - 0
            - payments_to_owners
            + in_from_st_investment
            - forecast.minimum_cash_required.loc[0]
            > 0
        ):
            lt_loan = (
                -(
                    0
                    + ncb_after_capex
                    + st_loan
                    - 0
                    - payments_to_owners
                    + in_from_st_investment
                    - forecast.minimum_cash_required.loc[0]
                )
                * policy.debt_financing_pct
            )

        ncb_financing_activities = st_loan + lt_loan - total_loan_payment

        return cls(
            years=years,
            inflows_from_sales=inflows_from_sales,
            total_operating_inflows=total_operating_inflows,
            payments_for_purchases=payments_for_purchases,
            admin_selling_expenses=admin_selling_expenses,
            income_taxes=income_taxes,
            total_operating_outflows=total_operating_outflows,
            operating_ncb=operating_ncb,
            investment_in_fixed_assets=investment_in_fixed_assets,
            ncb_investment_assets=ncb_investment_assets,
            ncb_after_capex=ncb_after_capex,
            st_loan_inflow=pd.Series([st_loan], index=years),
            lt_loan_inflow=pd.Series([lt_loan], index=years),
            principal_st_loan=st_loan_pp,
            interest_st_loan=st_loan_ip,
            total_st_loan_payment=st_loan_total,
            principal_lt_loan=lt_loan_pp,
            interest_lt_loan=lt_loan_ip,
            total_loan_payment=total_loan_payment,
            ncb_financing_activities=ncb_financing_activities,
        )

    def add_year(
        self,
        year: int,
        income_tax,
        sales_purchases,
        expenses,
        dep_and_investment,
    ):
        # WITH NEW CLASSES CHANGE
        # INCOME TAX !

        # compute single-year values
        admin_expense = expenses.total_as_expenses.loc[year]
        total_inflow = sales_purchases.total_inflows.loc[year]
        payment_for_purchases = sales_purchases.total_outflows.loc[year]
        invest = dep_and_investment.new_fixed_assets.loc[year]

        total_outflow = payment_for_purchases + admin_expense + income_tax
        oper_ncb = total_inflow - total_outflow
        ncb_inv = -invest
        ncb_after_capex = oper_ncb + ncb_inv

        # new index = old years + the new year
        self.years = self.years.append(pd.Index([year]))

        # use concat instead of deprecated Series.append
        self.inflows_from_sales = pd.concat(
            [
                self.inflows_from_sales,
                pd.Series([total_inflow], index=[year]),
            ]
        )
        self.total_operating_inflows = pd.concat(
            [self.total_operating_inflows, pd.Series([total_inflow], index=[year])]
        )
        self.payments_for_purchases = pd.concat(
            [
                self.payments_for_purchases,
                pd.Series([payment_for_purchases], index=[year]),
            ]
        )
        self.admin_selling_expenses = pd.concat(
            [self.admin_selling_expenses, pd.Series([admin_expense], index=[year])]
        )
        self.income_taxes = pd.concat(
            [self.income_taxes, pd.Series([income_tax], index=[year])]
        )
        self.total_operating_outflows = pd.concat(
            [self.total_operating_outflows, pd.Series([total_outflow], index=[year])]
        )
        self.operating_ncb = pd.concat(
            [self.operating_ncb, pd.Series([oper_ncb], index=[year])]
        )
        self.investment_in_fixed_assets = pd.concat(
            [self.investment_in_fixed_assets, pd.Series([invest], index=[year])]
        )
        self.ncb_investment_assets = pd.concat(
            [self.ncb_investment_assets, pd.Series([ncb_inv], index=[year])]
        )
        self.ncb_after_capex_series = pd.concat(
            [self.ncb_after_capex, pd.Series([ncb_after_capex], index=[year])]
        )

    # TODO maybe separate into ST and LT methods
    def add_financing(
        self,
        year,
        policy,
        forecast,
        transactions,
        loans,
    ):
        """
        Fill Table 9c once the borrowing schedule is known.
        """
        # Loans to take and pay or not

        # Loan payments first
        st_loan_pp = self.st_loan_inflow.loc[year - 1]
        st_loan_ip = loans.st_ip.loc[year]
        st_loan_total = st_loan_pp + st_loan_ip
        lt_loan_pp = loans.lt_total_pp.loc[year]
        lt_loan_ip = loans.lt_total_interest.loc[year]
        total_loan_payment = st_loan_total + lt_loan_pp + lt_loan_ip

        # Now take out loan?
        previous_ncb = (
            transactions.discretionary_transactions.year_ncb.loc[year - 1]
            if year > 0
            else 0.0
        )
        previous_cum_ncb = (
            transactions.discretionary_transactions.cum_ncb.loc[year - 1]
            if year > 0
            else 0.0
        )

        oper_ncb = self.operating_ncb.loc[year]

        st_loan_check = (
            previous_ncb
            + oper_ncb
            - st_loan_total
            - forecast.minimum_cash_required.loc[year]
        )
        st_loan = 0.0
        if st_loan_check > 0:
            st_loan = -st_loan_check

        payments_to_owners = transactions.owner_transactions.payments_to_owners.loc[
            year
        ]
        return_from_st_investment = (
            transactions.discretionary_transactions.return_from_st_investment.loc[year]
        )

        if (
            previous_cum_ncb
            + oper_ncb
            + st_loan
            - total_loan_payment
            - payments_to_owners
            + return_from_st_investment
            - forecast.minimum_cash_required.loc[year]
            > 0
        ):
            lt_loan = (
                -(
                    previous_cum_ncb
                    + oper_ncb
                    + st_loan
                    - total_loan_payment
                    - payments_to_owners
                    + return_from_st_investment
                    - forecast.minimum_cash_required.loc[year]
                )
                * policy.debt_financing_pct
            )

        ncb_financing_activities = st_loan + lt_loan - total_loan_payment

        # use concat instead of deprecated Series.append
        self.st_loan_inflow = pd.concat(
            [self.st_loan_inflow, pd.Series([st_loan], index=[year])]
        )
        self.lt_loan_inflow = pd.concat(
            [self.lt_loan_inflow, pd.Series([lt_loan], index=[year])]
        )
        self.principal_st_loan = pd.concat(
            [self.principal_st_loan, pd.Series([st_loan_pp], index=[year])]
        )
        self.interest_st_loan = pd.concat(
            [self.interest_st_loan, pd.Series([st_loan_ip], index=[year])]
        )
        self.total_st_loan_payment = pd.concat(
            [self.total_st_loan_payment, pd.Series([st_loan_total], index=[year])]
        )
        self.principal_lt_loan = pd.concat(
            [self.principal_lt_loan, pd.Series([lt_loan_pp], index=[year])]
        )
        self.interest_lt_loan = pd.concat(
            [self.interest_lt_loan, pd.Series([lt_loan_ip], index=[year])]
        )
        self.total_loan_payment = pd.concat(
            [self.total_loan_payment, pd.Series([total_loan_payment], index=[year])]
        )
        self.ncb_financing_activities = pd.concat(
            [
                self.ncb_financing_activities,
                pd.Series([ncb_financing_activities], index=[year]),
            ]
        )
