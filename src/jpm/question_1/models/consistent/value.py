from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DepreciationSchedule:
    years: pd.Index

    beginning_nfa: pd.Series

    dep_invest_year0: pd.DataFrame
    dep_invest_year1: pd.DataFrame
    dep_invest_year2: pd.DataFrame
    dep_invest_year3: pd.DataFrame

    annual_depreciation: pd.Series
    cumulated_depreciation: pd.Series

    investment_keep_constant: pd.Series
    investment_for_growth: pd.Series

    new_fixed_assets: pd.Series
    net_fixed_assets: pd.Series

    # @classmethod
    # def initial(cls, input_data) -> "DepreciationSchedule":
    #     beggining_nfa = pd.Series([0.0], index=[0])
    #     invest_to_keep = input_data.net_fixed_assets
    #     annual_dep = 0
    #     cum_dep = 0
    #     invest_to_grow = 0
    #     new_fa = invest_to_keep + invest_to_grow
    #     net_fa = beggining_nfa

    #     return cls(
    #         years=input_data.years,
    #         beginning_nfa=beggining_nfa,
    #         dep_invest=pd.DataFrame(),
    #         annual_depreciation=pd.Series([annual_dep], index=[0]),
    #         cumulated_depreciation=pd.Series([cum_dep], index=[0]),
    #         investment_keep_constant=pd.Series([invest_to_keep], index=[0]),
    #         investment_for_growth=pd.Series([invest_to_grow], index=[0]),
    #         new_fixed_assets=pd.Series([new_fa], index=[0]),
    #         net_fixed_assets=pd.Series([net_fa], index=[0]),
    #     )

    @classmethod
    def from_inputs(cls, input_data) -> "DepreciationSchedule":
        years = input_data.years
        T = len(years)

        life = int(round(input_data.lineal_depreciation.iloc[0]))
        initial_nfa = float(input_data.net_fixed_assets)

        beginning_nfa = [0.0] * T
        new_fa = [0.0] * T
        inv_const = [0.0] * T
        inv_growth = [0.0] * T
        annual_dep = [0.0] * T
        net_fa = [0.0] * T

        for t in range(T):
            # Row 66: Beginning NFA
            if t == 0:
                beginning_nfa[t] = 0.0
            else:
                beginning_nfa[t] = net_fa[t - 1]

            # Row 72: Annual depreciation in year t
            dep_t = 0.0
            for tau in range(t):  # sum contributions from all previous cohorts
                if (t >= tau + 1) and (t <= tau + life):
                    dep_t += new_fa[tau] / life
            annual_dep[t] = dep_t

            # Row 74: Investment to keep NFA constant
            if t == 0:
                inv_const[t] = initial_nfa  # =D6 in spreadsheet
            else:
                inv_const[t] = annual_dep[t]  # =E72, F72, ...

            # Row 75: Investment for growth (=C77*E23)
            if t == 0:
                inv_growth[t] = 0.0
            elif t < T - 1:
                inv_growth[t] = (
                    net_fa[t - 1] * input_data.increase_sales_volume.iloc[t + 1]
                )
            else:
                inv_growth[t] = (
                    0.0  # last year has no growth investment - out of horizon
                )

            # Row 76: New fixed assets (=D75+D74)
            new_fa[t] = inv_const[t] + inv_growth[t]

            # Row 77: Net fixed assets (=D66+D76-D72)
            net_fa[t] = beginning_nfa[t] + new_fa[t] - annual_dep[t]

        beginning_nfa = pd.Series(beginning_nfa, index=years)
        new_fa = pd.Series(new_fa, index=years)
        inv_const = pd.Series(inv_const, index=years)
        inv_growth = pd.Series(inv_growth, index=years)
        annual_dep = pd.Series(annual_dep, index=years)
        net_fa = pd.Series(net_fa, index=years)
        cum_dep = annual_dep.cumsum()

        # Shift inv_growth back 1 year to match spreadsheet logic
        # print(beginning_nfa)

        # print(inv_growth)
        # breakpoint()

        # Rows 67–70: depreciation by investment year (cohorts)
        def cohort_dep(invest_year: int) -> pd.Series:
            vals = []
            for t in range(T):
                if (t >= invest_year + 1) and (t <= invest_year + life):
                    vals.append(new_fa.iloc[invest_year] / life)
                else:
                    vals.append(0.0)
            return pd.Series(vals, index=years)

        dep_y0 = cohort_dep(0)  # “Annual depreciation for investment in year 0”
        dep_y1 = cohort_dep(1)  # “… in year 1”
        dep_y2 = cohort_dep(2)  # “… in year 2”
        dep_y3 = cohort_dep(3)  # “… in year 3”

        return cls(
            years=years,
            beginning_nfa=beginning_nfa,
            dep_invest_year0=dep_y0,
            dep_invest_year1=dep_y1,
            dep_invest_year2=dep_y2,
            dep_invest_year3=dep_y3,
            annual_depreciation=annual_dep,
            cumulated_depreciation=cum_dep,
            investment_keep_constant=inv_const,
            investment_for_growth=inv_growth,
            new_fixed_assets=new_fa,
            net_fixed_assets=net_fa,
        )


@dataclass(frozen=True)
class InventorySchedule:
    years: pd.Index

    # Units (rows 81–84)
    units_sold: pd.Series  # Row 81
    final_inventory: pd.Series  # Row 82 (units)
    initial_inventory: pd.Series  # Row 83 (units)
    purchases: pd.Series  # Row 84 (units)

    # Valuation (rows 90–94)
    unit_cost: pd.Series  # Row 90
    initial_inventory_value: pd.Series  # Row 91
    purchases_value: pd.Series  # Row 92
    final_inventory_value: pd.Series  # Row 93
    cogs: pd.Series  # Row 94

    @classmethod
    def from_inputs(
        cls,
        input_data,
        policy,
        forecast,
    ) -> "InventorySchedule":
        years = input_data.years
        T = len(years)

        # -------------------------------------------------------
        # UNITS SCHEDULE (InventoryPurchasesUnits)
        # -------------------------------------------------------

        # Row 81 – Units sold (from Table 3)
        units_sold = forecast.sales_units.copy()
        # print(units_sold)
        # Row 82 – Final inventory (units)
        # Year 1: =D10 (initial inventory in units)
        # Years t>1: =E81 * E29  (units_sold_t * inventory_pct_t)
        final_inv_vals = [0.0] * T
        for t, _y in enumerate(years):
            if t == 0:
                final_inv_vals[t] = input_data.initial_inventory
            else:
                final_inv_vals[t] = units_sold.iloc[t] * policy.inventory_pct.iloc[t]
        final_inventory = pd.Series(final_inv_vals, index=years)

        # Row 83 – Initial inventory (units)
        # Year 1: 0
        # Years t>1: previous year's final inventory
        init_inv_vals = [0.0] * T
        for t in range(1, T):
            init_inv_vals[t] = final_inventory.iloc[t - 1]
        initial_inventory = pd.Series(init_inv_vals, index=years)

        # Row 84 – Purchases (units)
        # = units_sold_t + final_inv_t - initial_inv_t
        purchases = units_sold + final_inventory - initial_inventory

        # -------------------------------------------------------
        # FIFO VALUATION (InventoryFIFOValuation)
        # -------------------------------------------------------

        # Row 90 – UNIT COST
        # Year 0: =D11  (initial purchase price)
        # Year t>0: previous * (1 + purchasing growth)
        unit_cost_vals = [0.0] * T
        unit_cost_vals[0] = input_data.initial_purchase_price
        for t in range(1, T):
            g_nom = forecast.nominal_purchasing.iloc[t]
            unit_cost_vals[t] = unit_cost_vals[t - 1] * (1.0 + g_nom)
        unit_cost = pd.Series(unit_cost_vals, index=years)

        # Row 92 – Purchases value
        # = purchases_units * unit_cost
        purchases_value = purchases * unit_cost

        # Row 93 – Final inventory value
        # = final_inventory_units * unit_cost
        final_inv_val = final_inventory * unit_cost

        # Row 91 – Initial inventory value
        # Year 0: 0.0
        # Year t>0: previous year's final inventory value
        init_inv_val = [0.0] * T
        for t in range(1, T):
            init_inv_val[t] = final_inv_val.iloc[t - 1]
        initial_inventory_value = pd.Series(init_inv_val, index=years)

        # Row 94 – COGS
        # = initial_inventory_value + purchases_value - final_inventory_value
        cogs = initial_inventory_value + purchases_value - final_inv_val

        return cls(
            years=years,
            units_sold=units_sold,
            final_inventory=final_inventory,
            initial_inventory=initial_inventory,
            purchases=purchases,
            unit_cost=unit_cost,
            initial_inventory_value=initial_inventory_value,
            purchases_value=purchases_value,
            final_inventory_value=final_inv_val,
            cogs=cogs,
        )


@dataclass(frozen=True)
class SalesPurchasesSchedule:
    years: pd.Index

    # --- Table 8a: Timing (Rows 103–110) ---

    # Row 103 – Total sales revenues
    total_sales_revenues: pd.Series

    # Row 104 – Inflow from current year
    inflow_from_current_year: pd.Series

    # Row 105 – Credit sales
    credit_sales: pd.Series

    # Row 106 – Payment in advance from customers
    advance_from_customers: pd.Series

    # Row 107 – Total purchases (in dollars)
    total_purchases: pd.Series

    # Row 108 – Purchases paid the same year
    purchases_paid_same_year: pd.Series

    # Row 109 – Purchases on credit
    purchases_on_credit: pd.Series

    # Row 110 – Payment in advance to suppliers
    advance_to_suppliers: pd.Series

    # --- Table 8b: Flows (Rows 113–120) ---

    # Row 113 – Sales revenues from current year
    sales_revenues_current_year: pd.Series

    # Row 114 – Accounts Receivable (flows)
    accounts_receivable_flow: pd.Series

    # Row 115 – Advance payments from customers (flows)
    advance_payments_from_customers: pd.Series

    # Row 116 – Total inflows
    total_inflows: pd.Series

    # Row 117 – Purchases paid the current year
    purchases_paid_current_year: pd.Series

    # Row 118 – Payment of Accounts Payable
    payment_accounts_payable: pd.Series

    # Row 119 – Advance payment to suppliers
    advance_payment_to_suppliers: pd.Series

    # Row 120 – Total outflows
    total_outflows: pd.Series

    @classmethod
    def from_inputs(
        cls,
        policy,
        forecast,
        inv,
    ) -> "SalesPurchasesSchedule":
        years = forecast.years

        # ---- Policy parameters (taken as constants from year 1 row) ----
        ar_rate = float(policy.ar_pct.iloc[1])  # E30
        adv_cust_rate = float(policy.adv_from_cust_pct.iloc[1])  # E31
        ap_rate = float(policy.ap_pct.iloc[1])  # E32
        adv_sup_rate = float(policy.adv_to_suppliers_pct.iloc[1])  # E33

        # --------------------------------------------------------
        # Table 8a – Timing (Rows 103–110)
        # --------------------------------------------------------

        # Row 103 – Total sales revenues
        total_sales = forecast.total_sales.copy()

        # Row 104 – Inflow from current year: sales * (1 - AR% - adv_cust%)
        inflow_current = total_sales * (1.0 - ar_rate - adv_cust_rate)
        # Year 0 is 0.0 in the sheet

        # Row 105 – Credit sales: sales * AR%
        credit_sales = total_sales * ar_rate

        # Row 106 – Payment in advance from customers: sales * adv_cust%
        payments_in_advance = total_sales * adv_cust_rate

        # Row 107 – Total purchases (value): from Table 6b
        total_purchases = inv.purchases_value.copy()

        # Row 108 – Purchases paid same year: purchases * (1 - AP% - adv_sup%)
        # If year 0, = total_purchases (no credit or advance payments yet)
        purchases_paid_same_year = total_purchases * (1.0 - ap_rate - adv_sup_rate)
        purchases_paid_same_year.iloc[0] = total_purchases.iloc[0]

        # Row 109 – Purchases on credit: purchases * AP%
        purchases_on_credit = total_purchases * ap_rate
        year0_credit = total_purchases.iloc[0] - purchases_paid_same_year.iloc[0]
        purchases_on_credit.iloc[0] = year0_credit

        # Row 110 – Payment in advance to suppliers: purchases * adv_sup%
        adv_to_suppliers = total_purchases * adv_sup_rate
        # shift a year
        adv_to_suppliers = adv_to_suppliers.shift(-1).fillna(0.0)

        # --------------------------------------------------------
        # Table 8b – Flows (Rows 113–120)
        # --------------------------------------------------------

        # Row 113 – Sales revenues from current year
        # = inflow_from_current_year (Row 104)
        sales_rev_current = inflow_current.copy()

        # Row 114 – Accounts Receivable (flows)
        # AR_flow[t] = credit_sales[t-1], AR_flow[0] = 0
        ar_vals = [0.0] * len(years)
        for t in range(1, len(years)):
            ar_vals[t] = credit_sales.iloc[t - 1]
        accounts_receivable_flow = pd.Series(ar_vals, index=years)

        # Row 115 – Advance payments from customers (flows)
        advance_payments_from_customers = payments_in_advance.copy()
        # Shifted a year ahead
        advance_payments_from_customers = advance_payments_from_customers.shift(
            -1
        ).fillna(0.0)

        # Row 116 – Total inflows
        total_inflows = (
            accounts_receivable_flow
            + sales_rev_current
            + advance_payments_from_customers
        )

        # Row 117 – Purchases paid the current year
        purchases_paid_current_year = purchases_paid_same_year.copy()

        # Row 118 – Payment of Accounts Payable
        # AP_flow[t] = purchases_on_credit[t-1], AP_flow[0] = 0
        ap_vals = [0.0] * len(years)
        for t in range(1, len(years)):
            ap_vals[t] = purchases_on_credit.iloc[t - 1]
        payment_accounts_payable = pd.Series(ap_vals, index=years)

        # Row 119 – Advance payment to suppliers
        advance_payment_to_suppliers = adv_to_suppliers.copy()

        # Row 120 – Total outflows
        total_outflows = (
            purchases_paid_current_year
            + payment_accounts_payable
            + advance_payment_to_suppliers
        )

        return cls(
            years=years,
            total_sales_revenues=total_sales,
            inflow_from_current_year=inflow_current,
            credit_sales=credit_sales,
            advance_from_customers=payments_in_advance,
            total_purchases=total_purchases,
            purchases_paid_same_year=purchases_paid_same_year,
            purchases_on_credit=purchases_on_credit,
            advance_to_suppliers=adv_to_suppliers,
            sales_revenues_current_year=sales_rev_current,
            accounts_receivable_flow=accounts_receivable_flow,
            advance_payments_from_customers=advance_payments_from_customers,
            total_inflows=total_inflows,
            purchases_paid_current_year=purchases_paid_current_year,
            payment_accounts_payable=payment_accounts_payable,
            advance_payment_to_suppliers=advance_payment_to_suppliers,
            total_outflows=total_outflows,
        )
