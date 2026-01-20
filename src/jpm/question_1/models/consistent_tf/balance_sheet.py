from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import tensorflow as tf

if TYPE_CHECKING:
    from income_statement import IncomeStatement
    from loans import LoanSchedules
    from trans import Transactions
    from value import DepreciationSchedule, InventorySchedule, SalesPurchasesSchedule


@dataclass(frozen=True)
class BalanceSheet:
    """Balance sheet assembled from transaction, loan, and inventory
    schedules (TensorFlow version)."""

    # shape: [T]
    year: tf.Tensor

    cash_cb: tf.Tensor
    ar_it: tf.Tensor
    inventory_it: tf.Tensor
    app_it: tf.Tensor
    st_investments_cb: tf.Tensor
    current_assets: tf.Tensor
    net_fixed_assets_it: tf.Tensor
    total_assets: tf.Tensor

    ap_it: tf.Tensor
    apr: tf.Tensor
    short_term_debt_cb: tf.Tensor
    current_liabilities: tf.Tensor
    long_term_debt_cb: tf.Tensor
    total_liabilities: tf.Tensor

    equity_investment_cb: tf.Tensor
    retained_earnings_is: tf.Tensor
    current_year_ni: tf.Tensor
    repurchase_of_equity: tf.Tensor

    total_liabilities_and_equity: tf.Tensor
    check: tf.Tensor

    @classmethod
    def from_inputs(
        cls,
        year,
        transactions: "Transactions",
        sales_purchases: "SalesPurchasesSchedule",
        inventory: "InventorySchedule",
        depreciation: "DepreciationSchedule",
        loanbook: "LoanSchedules",
        income_statement: "IncomeStatement",
        dtype: tf.dtypes.DType = tf.float32,
    ) -> "BalanceSheet":
        # years as int tensor (or keep as whatever you want, but tensor not Index)
        yr = tf.convert_to_tensor(year)
        # everything below is assumed 1-D, same length as years
        years = transactions.years
        years = years[: year - 1]
        # print(f"\nyear: {year}, years tensor: {years}\n")

        cash_cb = transactions.discretionary.cumulated_ncb.loc[year]

        ar_it = sales_purchases.credit_sales.loc[year]
        inventory_it = inventory.final_inventory_value.loc[year]
        app_it = sales_purchases.advance_payment_to_suppliers.loc[year]
        st_investments_cb = transactions.discretionary.st_investments.loc[year]
        net_fixed_assets_it = depreciation.net_fixed_assets.loc[year]
        current_assets = cash_cb + ar_it + inventory_it + app_it + st_investments_cb
        total_assets = current_assets + net_fixed_assets_it

        ap_it = sales_purchases.purchases_on_credit.loc[year]
        apr = sales_purchases.advance_payments_from_customers.loc[year]
        loan_summary = loanbook.book.schedule_summary(year)
        short_term_debt_cb = loan_summary.short_term.ending_balance
        long_term_debt_cb = loan_summary.long_term.ending_balance

        current_liabilities = ap_it + apr + short_term_debt_cb
        total_liabilities = current_liabilities + long_term_debt_cb

        equity_cb_vals = []
        eq_cb_prev = 0.0
        for y in range(0, year + 1):
            eq_cb_curr = eq_cb_prev + transactions.owner.invested_equity.loc[y]
            equity_cb_vals.append(eq_cb_curr)
            eq_cb_prev = eq_cb_curr

        equity_investment_cb = equity_cb_vals[-1]

        retained_earnings_is = income_statement.cre
        current_year_ni = tf.constant(income_statement.net_income, dtype=dtype)

        repurchase_vals = []
        rep_prev = 0.0
        for y in years:
            rep_curr = rep_prev - transactions.owner.repurchased_stock.loc[y]
            repurchase_vals.append(rep_curr)
            rep_prev = rep_curr
        repurchase_of_equity = tf.constant(repurchase_vals[-1], dtype=dtype)

        total_liabilities_and_equity = (
            total_liabilities
            + equity_investment_cb
            + retained_earnings_is
            + current_year_ni
            + repurchase_of_equity
        )

        check = total_liabilities_and_equity - total_assets

        return cls(
            year=yr,
            cash_cb=cash_cb,
            ar_it=ar_it,
            inventory_it=inventory_it,
            app_it=app_it,
            st_investments_cb=st_investments_cb,
            current_assets=current_assets,
            net_fixed_assets_it=net_fixed_assets_it,
            total_assets=total_assets,
            ap_it=ap_it,
            apr=apr,
            short_term_debt_cb=short_term_debt_cb,
            current_liabilities=current_liabilities,
            long_term_debt_cb=long_term_debt_cb,
            total_liabilities=total_liabilities,
            equity_investment_cb=equity_investment_cb,
            retained_earnings_is=retained_earnings_is,
            current_year_ni=current_year_ni,
            repurchase_of_equity=repurchase_of_equity,
            total_liabilities_and_equity=total_liabilities_and_equity,
            check=check,
        )

    @staticmethod
    def _to_float(x: Any, idx: Optional[int] = None) -> float:
        """
        Convert a tf.Tensor (scalar or 1D) or numeric value to a Python float.

        - Scalars: ignore idx
        - 1D tensors/arrays: require idx (e.g. -1 for last year)
        """
        if isinstance(x, tf.Tensor):
            arr = x.numpy()
        else:
            # Plain Python/NumPy scalar
            return float(x)

        if arr.shape == ():  # scalar
            return float(arr.item())

        if idx is None:
            raise ValueError("idx is required for non-scalar fields")

        return float(arr[idx])

    def pretty_print(self, idx: int = -1) -> None:
        """
        Nicely print this BalanceSheet instance for a single year (chosen by
        idx for vector fields like equity_investment_cb,
        total_liabilities_and_equity, etc.).

        Parameters
        ----------
        idx : int, default -1
            Index into 1D fields (e.g. last element for year 4 if you have 0..4).
        """
        # Scalars
        year = BalanceSheet._to_float(self.year)
        cash = BalanceSheet._to_float(self.cash_cb)
        ar = BalanceSheet._to_float(self.ar_it)
        inventory = BalanceSheet._to_float(self.inventory_it)
        app = BalanceSheet._to_float(self.app_it)
        st_inv = BalanceSheet._to_float(self.st_investments_cb)
        current_assets = BalanceSheet._to_float(self.current_assets)
        net_fixed_assets = BalanceSheet._to_float(self.net_fixed_assets_it)
        total_assets = BalanceSheet._to_float(self.total_assets)

        ap = BalanceSheet._to_float(self.ap_it)
        apr = BalanceSheet._to_float(self.apr)
        std = BalanceSheet._to_float(self.short_term_debt_cb)
        current_liabilities = BalanceSheet._to_float(self.current_liabilities)
        ltd = BalanceSheet._to_float(self.long_term_debt_cb)
        total_liabilities = BalanceSheet._to_float(self.total_liabilities)

        current_year_ni = BalanceSheet._to_float(self.current_year_ni)

        # Vector fields – take a specific index
        equity_investment = BalanceSheet._to_float(self.equity_investment_cb, idx)
        retained_earnings = BalanceSheet._to_float(
            self.retained_earnings_is, 0
        )  # shape (1,) in your printout
        repurchase = BalanceSheet._to_float(self.repurchase_of_equity, idx)
        total_liab_and_equity = BalanceSheet._to_float(
            self.total_liabilities_and_equity, idx
        )
        check_val = BalanceSheet._to_float(self.check, idx)

        # Derived equity (just for clarity)
        equity_from_components = equity_investment + retained_earnings - repurchase
        implied_equity_from_total = total_liab_and_equity - total_liabilities

        imbalance = total_assets - total_liab_and_equity

        lines = []
        lines.append(f"Balance Sheet (year {int(year)}, idx={idx})")
        lines.append("-" * 60)

        # Assets
        lines.append("ASSETS")
        lines.append("  Current assets:")
        lines.append(f"    Cash                         {cash:10.4f}")
        lines.append(f"    Accounts receivable          {ar:10.4f}")
        lines.append(f"    Inventory                    {inventory:10.4f}")
        lines.append(f"    Other current (APP)          {app:10.4f}")
        lines.append(f"    Short-term investments       {st_inv:10.4f}")
        lines.append(f"    -> Current assets (given)    {current_assets:10.4f}")
        lines.append("")
        lines.append("  Non-current assets:")
        lines.append(f"    Net fixed assets             {net_fixed_assets:10.4f}")
        lines.append("")
        lines.append(f"  TOTAL ASSETS                   {total_assets:10.4f}")
        lines.append("")

        # Liabilities
        lines.append("LIABILITIES")
        lines.append("  Current liabilities:")
        lines.append(f"    Accounts payable             {ap:10.4f}")
        lines.append(f"    APR                          {apr:10.4f}")
        lines.append(f"    Short-term debt              {std:10.4f}")
        lines.append(f"    -> Current liabilities       {current_liabilities:10.4f}")
        lines.append("")
        lines.append("  Long-term liabilities:")
        lines.append(f"    Long-term debt               {ltd:10.4f}")
        lines.append("")
        lines.append(f"  TOTAL LIABILITIES              {total_liabilities:10.4f}")
        lines.append("")

        # Equity
        lines.append("EQUITY")
        lines.append(f"  Equity investment[idx]         {equity_investment:10.4f}")
        lines.append(f"  Retained earnings              {retained_earnings:10.4f}")
        lines.append(f"  Repurchase of equity[idx]      {repurchase:10.4f}")
        lines.append(f"  -> Equity (components)         {equity_from_components:10.4f}")
        lines.append("")
        lines.append(f"  Liab + Equity (given)[idx]     {total_liab_and_equity:10.4f}")
        lines.append(
            f"  Implied equity = L+E - L       {implied_equity_from_total:10.4f}"
        )
        lines.append("")

        # Check / imbalance
        lines.append("CHECKS")
        lines.append(f"  Assets - (Liabilities + Equity)[idx] = {imbalance:10.4f}")
        lines.append(f"  check[idx] field                      = {check_val:10.4f}")
        lines.append(
            f"  Current year NI                       = {current_year_ni:10.4f}"
        )

        print("\n".join(lines))


def _to_float(x: Any, idx: Optional[int] = None) -> float:
    """
    Convert a tf.Tensor (scalar or 1D) or numeric value to a Python float.

    - Scalars: ignore idx
    - 1D tensors/arrays: require idx (e.g. -1 for last year)
    """
    if isinstance(x, tf.Tensor):
        arr = x.numpy()
    else:
        # Plain Python/NumPy scalar
        return float(x)

    if arr.shape == ():  # scalar
        return float(arr.item())

    if idx is None:
        raise ValueError("idx is required for non-scalar fields")

    return float(arr[idx])


def pretty_print_balance_sheet(bs, idx: int = -1) -> None:
    """
    Nicely print a BalanceSheet instance for a single year (chosen by idx
    for vector fields like equity_investment_cb,
    total_liabilities_and_equity, etc.).

    Parameters
    ----------
    bs : BalanceSheet
        Your BalanceSheet instance.
    idx : int, default -1
        Index into 1D fields (e.g. last element for year 4 if you have 0..4).
    """
    # Scalars
    year = _to_float(bs.year)
    cash = _to_float(bs.cash_cb)
    ar = _to_float(bs.ar_it)
    inventory = _to_float(bs.inventory_it)
    app = _to_float(bs.app_it)
    st_inv = _to_float(bs.st_investments_cb)
    current_assets = _to_float(bs.current_assets)
    net_fixed_assets = _to_float(bs.net_fixed_assets_it)
    total_assets = _to_float(bs.total_assets)

    ap = _to_float(bs.ap_it)
    apr = _to_float(bs.apr)
    std = _to_float(bs.short_term_debt_cb)
    current_liabilities = _to_float(bs.current_liabilities)
    ltd = _to_float(bs.long_term_debt_cb)
    total_liabilities = _to_float(bs.total_liabilities)

    current_year_ni = _to_float(bs.current_year_ni)

    # Vector fields – take a specific index
    equity_investment = _to_float(bs.equity_investment_cb, idx)
    retained_earnings = _to_float(
        bs.retained_earnings_is, 0
    )  # shape (1,) in your printout
    repurchase = _to_float(bs.repurchase_of_equity, idx)
    total_liab_and_equity = _to_float(bs.total_liabilities_and_equity, idx)
    check_val = _to_float(bs.check, idx)

    # Derived equity (just for clarity)
    equity_from_components = equity_investment + retained_earnings - repurchase
    implied_equity_from_total = total_liab_and_equity - total_liabilities

    imbalance = total_assets - total_liab_and_equity

    lines = []
    lines.append(f"Balance Sheet (year {int(year)}, idx={idx})")
    lines.append("-" * 60)

    # Assets
    lines.append("ASSETS")
    lines.append("  Current assets:")
    lines.append(f"    Cash                         {cash:10.4f}")
    lines.append(f"    Accounts receivable          {ar:10.4f}")
    lines.append(f"    Inventory                    {inventory:10.4f}")
    lines.append(f"    Other current (APP)          {app:10.4f}")
    lines.append(f"    Short-term investments       {st_inv:10.4f}")
    lines.append(f"    -> Current assets (given)    {current_assets:10.4f}")
    lines.append("")
    lines.append("  Non-current assets:")
    lines.append(f"    Net fixed assets             {net_fixed_assets:10.4f}")
    lines.append("")
    lines.append(f"  TOTAL ASSETS                   {total_assets:10.4f}")
    lines.append("")

    # Liabilities
    lines.append("LIABILITIES")
    lines.append("  Current liabilities:")
    lines.append(f"    Accounts payable             {ap:10.4f}")
    lines.append(f"    APR                          {apr:10.4f}")
    lines.append(f"    Short-term debt              {std:10.4f}")
    lines.append(f"    -> Current liabilities       {current_liabilities:10.4f}")
    lines.append("")
    lines.append("  Long-term liabilities:")
    lines.append(f"    Long-term debt               {ltd:10.4f}")
    lines.append("")
    lines.append(f"  TOTAL LIABILITIES              {total_liabilities:10.4f}")
    lines.append("")

    # Equity
    lines.append("EQUITY")
    lines.append(f"  Equity investment[idx]         {equity_investment:10.4f}")
    lines.append(f"  Retained earnings              {retained_earnings:10.4f}")
    lines.append(f"  Repurchase of equity[idx]      {repurchase:10.4f}")
    lines.append(f"  -> Equity (components)         {equity_from_components:10.4f}")
    lines.append("")
    lines.append(f"  Liab + Equity (given)[idx]     {total_liab_and_equity:10.4f}")
    lines.append(f"  Implied equity = L+E - L       {implied_equity_from_total:10.4f}")
    lines.append("")

    # Check / imbalance
    lines.append("CHECKS")
    lines.append(f"  Assets - (Liabilities + Equity)[idx] = {imbalance:10.4f}")
    lines.append(f"  check[idx] field                      = {check_val:10.4f}")
    lines.append(f"  Current year NI                       = {current_year_ni:10.4f}")

    print("\n".join(lines))
