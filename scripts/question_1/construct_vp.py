import pandas as pd

from jpm.question_1.models.consistent import (
    AdminSellingExpenses,
    BalanceSheet,
    CashBudget,
    CashFlow,
    DepreciationSchedule,
    DiscretionaryTransactions,
    Forecasting,
    IncomeStatement,
    InputData,
    InventorySchedule,
    LoanSchedules,
    MarketResearchInput,
    OwnerTransactions,
    PolicyTable,
    SalesPurchasesSchedule,
    Transactions,
)

years = pd.Index([0, 1, 2, 3, 4])

input_data = InputData(
    years=years,
    net_fixed_assets=45.0,
    lineal_depreciation=pd.Series([4.0, None, None, None, None], index=years),
    corporate_tax_rate=0.35,
    initial_inventory=4.0,
    initial_purchase_price=5.0,
    estimated_overhead_expenses=22.0,
    admin_and_sales_payroll=24.0,
    lt_years_loan_3=10,
    st_years_loan_2=1,
    inflation_rate=pd.Series([0.06, 0.06, 0.055, 0.055, 0.05], index=years),
    real_increase_selling_price=pd.Series([0.0, 0.01, 0.01, 0.01, 0.01], index=years),
    real_increase_purchase_price=pd.Series(
        [0.0, 0.005, 0.005, 0.005, 0.01], index=years
    ),
    real_increase_overheads=pd.Series([0.0, 0.005, 0.005, 0.005, 0.005], index=years),
    real_increase_payroll=pd.Series([0.0, 0.015, 0.015, 0.015, 0.015], index=years),
    increase_sales_volume=pd.Series([0.0, 0.0, 0.01, 0.02, 0.02], index=years),
    real_interest_rate=pd.Series([0, 0.02, 0.02, 0.02, 0.02], index=years),
    risk_premium_debt_cost=0.05,
    risk_premium_return_st_inv=-0.02,
)

market_research = MarketResearchInput(
    selling_price=7.0,
    elasticity_b=-0.350,
    elasticity_coef=100.0,
)

policy_table = PolicyTable(
    years=years,
    promo_ad=pd.Series([0.0, 0.03, 0.0, 0.0, 0.0], index=[0, 1, 2, 3, 4]),
    inventory_pct=pd.Series(
        [0.0, 1 / 12, 1 / 12, 1 / 12, 1 / 12], index=[0, 1, 2, 3, 4]
    ),
    ar_pct=pd.Series([0.0, 0.05, 0.05, 0.05, 0.05], index=[0, 1, 2, 3, 4]),
    adv_from_cust_pct=pd.Series([0.0, 0.10, 0.10, 0.10, 0.10], index=[0, 1, 2, 3, 4]),
    ap_pct=pd.Series([0.0, 0.10, 0.10, 0.10, 0.10], index=[0, 1, 2, 3, 4]),
    adv_to_suppliers_pct=pd.Series(
        [0.0, 0.10, 0.10, 0.10, 0.10], index=[0, 1, 2, 3, 4]
    ),
    payout_ratio=pd.Series([0.0, 0.70, 0.70, 0.70, 0.70], index=[0, 1, 2, 3, 4]),
    cash_pct_of_sales=pd.Series([0.0, 0.04, 0.04, 0.04, 0.04], index=[0, 1, 2, 3, 4]),
    debt_financing_pct=0.70,
    minimum_initial_cash=13.0,
    selling_commission_pct=pd.Series(
        [0.0, 0.04, 0.04, 0.04, 0.04], index=[0, 1, 2, 3, 4]
    ),
    stock_repurchase_pct=pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=[0, 1, 2, 3, 4]),
)

forecast = Forecasting.from_inputs(
    input_data=input_data,
    market_research=market_research,
    policy=policy_table,
)

depreciation = DepreciationSchedule.from_inputs(
    input_data=input_data,
)

inventory = InventorySchedule.from_inputs(
    input_data=input_data,
    policy=policy_table,
    forecast=forecast,
)

admin_expenses = AdminSellingExpenses.from_inputs(
    input_data=input_data,
    policy=policy_table,
    forecasts=forecast,
)

sales_purch_schedule = SalesPurchasesSchedule.from_inputs(
    policy=policy_table,
    forecast=forecast,
    inv=inventory,
)

loans = LoanSchedules()

cash_budget = CashBudget.initial(
    input_data=input_data,
    policy=policy_table,
    sales_purch_schedule=sales_purch_schedule,
    admin_selling_expenses=admin_expenses,
    dep_and_investment=depreciation,
    forecast=forecast,
    loans=loans,
)

owner_tx = OwnerTransactions.initial(
    policy=policy_table, cash_budget=cash_budget, depreciation=depreciation
)
discretionary_tx = DiscretionaryTransactions.from_inputs(
    policy=policy_table, cash_budget=cash_budget, owner_tx=owner_tx, forecast=forecast
)
transactions = Transactions(
    years=input_data.years,
    owners=owner_tx,
    discretionary=discretionary_tx,
    forecasting=forecast,
)

income_s = IncomeStatement.init_year(
    input_data=input_data,
    policy=policy_table,
    forecast_sales=forecast,
    inv_fifo=inventory,
    admin_selling=admin_expenses,
    depr_sched=depreciation,
    loan_schedules=loans,
)

for yr in years[1:]:
    income_s.add_year(
        year=yr,
        input_data=input_data,
        policy=policy_table,
        forecast_sales=forecast,
        inv_fifo=inventory,
        admin_selling=admin_expenses,
        depr_sched=depreciation,
        loan_schedules=loans,
    )

    cash_budget.add_year(
        year=yr,
        income_statement=income_s,
        sales_purchases=sales_purch_schedule,
        expenses=admin_expenses,
        dep_and_investment=depreciation,
        forecast=forecast,
        transactions=transactions,
        loans=loans,
        input_data=input_data,
        policy=policy_table,
    )

    transactions.add_year(yr, forecast, policy_table, cash_budget)

bs = BalanceSheet.from_inputs(
    years=input_data.years,
    transactions=transactions,
    sales_purchases=sales_purch_schedule,
    inventory=inventory,
    depreciation=depreciation,
    loanbook=loans,
    income_statement=income_s,
)

cf = CashFlow.from_inputs(
    years=input_data.years,
    input_data=input_data,
    cash_budget=cash_budget,
    transactions=transactions,
    income_statement=income_s,
)


bs_df = pd.concat(
    [
        series.rename(field_name)
        for field_name in bs.__dataclass_fields__
        if isinstance(series := getattr(bs, field_name), pd.Series)
    ],
    axis=1,
)
bs_df.index.name = "year"
print("Balance Sheet:")
print(bs_df.T.round(1))

cf_df = pd.concat(
    [
        series.rename(field_name)
        for field_name in cf.__dataclass_fields__
        if isinstance(series := getattr(cf, field_name), pd.Series)
    ],
    axis=1,
)
cf_df.index.name = "year"
print("\nCash Flow Statement:")
print(cf_df.T.round(1))
