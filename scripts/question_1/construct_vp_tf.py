from jpm.question_1.models.consistent_tf import (
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
    ValuationInputs,
)

years = [0, 1, 2, 3, 4]

input_data = InputData(
    years=years,
    net_fixed_assets=45.0,
    lineal_depreciation=4.0,
    corporate_tax_rate=0.35,
    initial_inventory=4.0,
    initial_purchase_price=5.0,
    estimated_overhead_expenses=22.0,
    admin_and_sales_payroll=24.0,
    lt_years_loan_3=10,
    st_years_loan_2=1,
    inflation_rate=[0.06, 0.06, 0.055, 0.055, 0.05],
    real_increase_selling_price=[0.0, 0.01, 0.01, 0.01, 0.01],
    real_increase_purchase_price=[0.0, 0.005, 0.005, 0.005, 0.01],
    real_increase_overheads=[0.0, 0.005, 0.005, 0.005, 0.005],
    real_increase_payroll=[0.0, 0.015, 0.015, 0.015, 0.015],
    increase_sales_volume=[0.0, 0.0, 0.01, 0.02, 0.02],
    real_interest_rate=[0, 0.02, 0.02, 0.02, 0.02],
    risk_premium_debt_cost=0.05,
    risk_premium_return_st_inv=-0.02,
    observed_kk=0.15,
    perpetual_leverage=0.3,
    expected_inflation_rate=0.0,
    real_growth_rate=0.0,
)

market_research = MarketResearchInput(
    selling_price=7.0,
    elasticity_b=-0.350,
    elasticity_coef=100.0,
)

policy_table = PolicyTable(
    years=years,
    promo_ad=[0.0, 0.03, 0.0, 0.0, 0.0],
    inventory_pct=[0.0, 1 / 12, 1 / 12, 1 / 12, 1 / 12],
    ar_pct=[0.0, 0.05, 0.05, 0.05, 0.05],
    adv_from_cust_pct=[0.0, 0.10, 0.10, 0.10, 0.10],
    ap_pct=[0.0, 0.10, 0.10, 0.10, 0.10],
    adv_to_suppliers_pct=[0.0, 0.10, 0.10, 0.10, 0.10],
    payout_ratio=[0.0, 0.70, 0.70, 0.70, 0.70],
    cash_pct_of_sales=[0.0, 0.04, 0.04, 0.04, 0.04],
    debt_financing_pct=0.70,
    minimum_initial_cash=13.0,
    selling_commission_pct=[0.0, 0.04, 0.04, 0.04, 0.04],
    stock_repurchase_pct=[0.0, 0.0, 0.0, 0.0, 0.0],
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
previous_is = income_s


for yr in years[1:]:
    income_s = income_s.add_year(
        year=yr,
        input_data=input_data,
        policy=policy_table,
        forecast_sales=forecast,
        inv_fifo=inventory,
        admin_selling=admin_expenses,
        depr_sched=depreciation,
        loan_schedules=loans,
        prev_is=income_s,
    )

    cash_budget = cash_budget.add_year(
        year=yr,
        income_statement=income_s,
        previous_is=previous_is,
        sales_purchases=sales_purch_schedule,
        expenses=admin_expenses,
        dep_and_investment=depreciation,
        forecast=forecast,
        transactions=transactions,
        loans=loans,
        input_data=input_data,
        policy=policy_table,
        previous_cash=cash_budget,
    )

    transactions.add_year(yr, forecast, policy_table, cash_budget)

    # Check this is being set in the correct place
    previous_is = income_s

    # Create Financial Statements at each step now
    if yr > 1:
        bs = BalanceSheet.from_inputs(
            year=yr,
            transactions=transactions,
            sales_purchases=sales_purch_schedule,
            inventory=inventory,
            depreciation=depreciation,
            loanbook=loans,
            income_statement=income_s,
        )
        bs.pretty_print()

        cf = CashFlow.from_inputs(
            years=input_data.years,
            input_data=input_data,
            cash_budget=cash_budget,
            transactions=transactions,
            income_statement=income_s,
        )
        cf.pretty_print()

val_inputs = ValuationInputs.from_inputs(year=-1, input_data=input_data)
tv_val = val_inputs.tv_and_liquidate(bs, income_s)
print(f"Val: {tv_val}")
