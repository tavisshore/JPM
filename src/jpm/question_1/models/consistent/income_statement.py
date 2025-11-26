from dataclasses import dataclass

import pandas as pd


@dataclass
class IncomeStatement:
    years: pd.Index

    sales_revenues: pd.Series  # Row 196
    cogs: pd.Series  # Row 197
    gross_income: pd.Series  # Row 198
    admin_selling_expenses: pd.Series  # Row 199
    depreciation: pd.Series  # Row 200
    ebit: pd.Series  # Row 201
    interest_payments: pd.Series  # Row 202
    return_from_st_investment: pd.Series  # Row 203
    ebt: pd.Series  # Row 204
    income_taxes: pd.Series  # Row 205
    net_income: pd.Series  # Row 206
    next_year_dividends: pd.Series  # Row 207
    cre: pd.Series  # Row 208 – Cumulated Retained Earnings

    @classmethod
    def init_year(
        cls,
        input_data,
        policy,
        forecast_sales,  # Table 3
        inv_fifo,  # Table 6b
        admin_selling,  # Table 7
        depr_sched,  # Table 5
        loan_schedules,  # Tables 11a/11b
    ) -> "IncomeStatement":
        full_years = input_data.years
        y0 = full_years[0]
        years = pd.Index([y0])

        sales_revenues = pd.Series([forecast_sales.total_sales.loc[y0]], index=years)
        cogs = pd.Series([inv_fifo.cogs.loc[y0]], index=years)
        gross_income = sales_revenues - cogs
        admin_selling_expenses = pd.Series(
            [admin_selling.total_as_expenses.loc[y0]], index=years
        )
        depreciation = pd.Series([depr_sched.annual_depreciation.loc[y0]], index=years)
        ebit = gross_income - admin_selling_expenses - depreciation
        dp = loan_schedules.debt_payments(y0)
        interest_payments = pd.Series([dp.st_interest + dp.lt_interest], index=years)
        return_from_st_investment = pd.Series([0.0], index=years)
        ebt = ebit + return_from_st_investment - interest_payments
        tax_rate = input_data.corporate_tax_rate
        income_taxes = ebt.clip(lower=0.0) * tax_rate
        net_income = ebt - income_taxes
        payout_ratio = float(policy.payout_ratio.iloc[1])
        next_year_dividends = net_income * payout_ratio
        cre = pd.Series([net_income.iloc[0] - next_year_dividends.iloc[0]], index=years)

        return cls(
            years=years,
            sales_revenues=sales_revenues,
            cogs=cogs,
            gross_income=gross_income,
            admin_selling_expenses=admin_selling_expenses,
            depreciation=depreciation,
            ebit=ebit,
            interest_payments=interest_payments,
            return_from_st_investment=return_from_st_investment,
            ebt=ebt,
            income_taxes=income_taxes,
            net_income=net_income,
            next_year_dividends=next_year_dividends,
            cre=cre,
        )

    def add_year(
        self,
        year,
        input_data,
        policy,
        forecast_sales,  # Table 3
        inv_fifo,  # Table 6b
        admin_selling,  # Table 7
        depr_sched,  # Table 5
        loan_schedules,  # Tables 11a/11b
    ):
        # New index with added year
        self.years = self.years.append(pd.Index([year]))

        sales_revenues_t = forecast_sales.total_sales.loc[year]
        cogs_t = inv_fifo.cogs.loc[year]
        gross_income_t = sales_revenues_t - cogs_t
        admin_selling_expenses_t = admin_selling.total_as_expenses.loc[year]
        depreciation_t = depr_sched.annual_depreciation.loc[year]
        ebit_t = gross_income_t - admin_selling_expenses_t - depreciation_t
        dp = loan_schedules.debt_payments(year)
        interest_payments_t = dp.st_interest + dp.lt_interest
        return_from_st_investment_t = 0.0  # assumed zero for year 0
        ebt_t = ebit_t + return_from_st_investment_t - interest_payments_t
        tax_rate = input_data.corporate_tax_rate
        income_taxes_t = max(0.0, ebt_t) * tax_rate
        net_income_t = ebt_t - income_taxes_t
        payout_ratio = float(policy.payout_ratio.iloc[1])
        next_year_dividends_t = net_income_t * payout_ratio

        cre_prev = float(self.cre.iloc[-1])
        cre_t = cre_prev + net_income_t - next_year_dividends_t
        # Append new values to existing series
        self.sales_revenues = pd.concat(
            [self.sales_revenues, pd.Series([sales_revenues_t], index=[year])]
        )
        self.cogs = pd.concat([self.cogs, pd.Series([cogs_t], index=[year])])
        self.gross_income = pd.concat(
            [self.gross_income, pd.Series([gross_income_t], index=[year])]
        )
        self.admin_selling_expenses = pd.concat(
            [
                self.admin_selling_expenses,
                pd.Series([admin_selling_expenses_t], index=[year]),
            ]
        )
        self.depreciation = pd.concat(
            [self.depreciation, pd.Series([depreciation_t], index=[year])]
        )
        self.ebit = pd.concat([self.ebit, pd.Series([ebit_t], index=[year])])
        self.interest_payments = pd.concat(
            [self.interest_payments, pd.Series([interest_payments_t], index=[year])]
        )
        self.return_from_st_investment = pd.concat(
            [
                self.return_from_st_investment,
                pd.Series([return_from_st_investment_t], index=[year]),
            ]
        )
        self.ebt = pd.concat([self.ebt, pd.Series([ebt_t], index=[year])])
        self.income_taxes = pd.concat(
            [self.income_taxes, pd.Series([income_taxes_t], index=[year])]
        )
        self.net_income = pd.concat(
            [self.net_income, pd.Series([net_income_t], index=[year])]
        )
        self.next_year_dividends = pd.concat(
            [self.next_year_dividends, pd.Series([next_year_dividends_t], index=[year])]
        )
        self.cre = pd.concat([self.cre, pd.Series([cre_t], index=[year])])
