from __future__ import annotations

import os
from typing import List


def count_leaf_list_values(d):
    count = 0
    for value in d.values():
        if isinstance(value, dict):
            count += count_leaf_list_values(value)
        elif isinstance(value, list):
            count += len(value)
    return count


def leaf_list_values(d):
    values = []
    for value in d.values():
        if isinstance(value, dict):
            values.extend(leaf_list_values(value))
        elif isinstance(value, list):
            values.extend(value)
    return values


def _is_income_before_taxes_feature(feature: str) -> bool:
    """Check if feature represents income before taxes."""
    lower = feature.lower()
    return (
        "before income taxes" in lower or "before taxes" in lower or "pretax" in lower
    )


def _fix_income_before_taxes(mapped_structure: dict, unmapped: list) -> None:
    """Fix Income Before Taxes classification."""
    for feature in list(unmapped):
        if _is_income_before_taxes_feature(feature):
            unmapped.remove(feature)
            if "Income Before Taxes" not in mapped_structure:
                mapped_structure["Income Before Taxes"] = []
            if not isinstance(mapped_structure.get("Income Before Taxes"), list):
                mapped_structure["Income Before Taxes"] = []
            mapped_structure["Income Before Taxes"].append(feature)
            print(f"✓ Post-processing fix: '{feature[:60]}...' → Income Before Taxes")


def _dedupe_income_before_taxes(mapped_structure: dict) -> None:
    """Deduplicate Income Before Taxes features."""
    ibt_features = mapped_structure.get("Income Before Taxes", [])
    if len(ibt_features) > 1:
        ibt_features_sorted = sorted(ibt_features, key=len)
        kept_feature = ibt_features_sorted[0]
        removed_features = ibt_features_sorted[1:]

        mapped_structure["Income Before Taxes"] = [kept_feature]
        mapped_structure.setdefault("__unmapped__", []).extend(removed_features)

        print("⚠ Warning: Multiple 'Income Before Taxes' features detected")
        print(f"  Kept: '{kept_feature}'")
        for rf in removed_features:
            print(f"  Moved to unmapped: '{rf}'")


def _fix_nonoperating_items(mapped_structure: dict, unmapped: list) -> None:
    """Fix Non-Operating Items classification."""
    for feature in list(unmapped):
        if feature.lower() == "nonoperating income expense":
            unmapped.remove(feature)
            if "Non-Operating Items" in mapped_structure:
                non_op = mapped_structure["Non-Operating Items"]
                if not isinstance(non_op.get("Total Non-Operating Items"), list):
                    non_op["Total Non-Operating Items"] = []
                non_op["Total Non-Operating Items"].append(feature)
                print(f"✓ Post-processing fix: '{feature}' → Total Non-Operating Items")


def _fix_nonop_duplicates(mapped_structure: dict) -> None:
    """Fix duplicates in Non-Operating Items."""
    if "Non-Operating Items" not in mapped_structure:
        return

    total_nonop = mapped_structure["Non-Operating Items"].get(
        "Total Non-Operating Items", []
    )
    other_nonop = mapped_structure["Non-Operating Items"].get(
        "Other Income (Expense)", []
    )

    if len(total_nonop) > 1:
        for feature in list(total_nonop):
            if "other nonoperating" in feature.lower() and len(total_nonop) > 1:
                total_nonop.remove(feature)
                other_nonop.append(feature)
                msg = f"✓ Post-processing fix: '{feature}' moved from "
                msg += "Total Non-Op → Other Income (Expense)"
                print(msg)


def _fix_income_statement(mapped_structure: dict) -> dict:
    """Apply fixes for income statement edge cases."""
    unmapped = mapped_structure.get("__unmapped__", [])
    _fix_income_before_taxes(mapped_structure, unmapped)
    _dedupe_income_before_taxes(mapped_structure)
    _fix_nonoperating_items(mapped_structure, unmapped)
    _fix_nonop_duplicates(mapped_structure)
    return mapped_structure


def _fix_balance_sheet(mapped_structure: dict) -> dict:
    """Apply fixes for balance sheet edge cases."""
    unmapped = mapped_structure.get("__unmapped__", [])

    for feature in list(unmapped):
        if "shares" in feature.lower() and (
            "issued" in feature.lower() or "outstanding" in feature.lower()
        ):
            print(f"✓ Verified unmapped share count: '{feature}'")

    return mapped_structure


def _fix_acquisitions_net_of_cash(mapped_structure: dict, unmapped: list) -> None:
    """Fix acquisitions net of cash acquired classification."""
    for feature in list(unmapped):
        if "acquisitions net of cash acquired" in feature.lower():
            unmapped.remove(feature)
            if "Investing Activities" in mapped_structure:
                inv = mapped_structure["Investing Activities"]
                acq_key = "Acquisitions (net of cash acquired)"
                if not isinstance(inv.get(acq_key), list):
                    inv[acq_key] = []
                inv[acq_key].append(feature)
                print(f"✓ Post-processing fix: '{feature[:60]}...' → Acquisitions")


def _is_net_change_in_cash(feature: str) -> bool:
    """Check if feature represents net change in cash."""
    lower = feature.lower()
    return ("cash" in lower and "period increase decrease" in lower) or (
        "cash equivalents" in lower and "increase decrease" in lower
    )


def _fix_net_change_in_cash(mapped_structure: dict, unmapped: list) -> None:
    """Fix net change in cash classification."""
    for feature in list(unmapped):
        if _is_net_change_in_cash(feature):
            unmapped.remove(feature)
            if not isinstance(mapped_structure.get("Net Change in Cash"), list):
                mapped_structure["Net Change in Cash"] = []
            mapped_structure["Net Change in Cash"].append(feature)
            print(f"✓ Post-processing fix: '{feature[:60]}...' → Net Change in Cash")


def _fix_payments_to_acquire(mapped_structure: dict, unmapped: list) -> None:
    """Fix payments to acquire businesses classification."""
    for feature in list(unmapped):
        if "payments to acquire businesses" not in feature.lower():
            continue

        acquisitions = mapped_structure["Investing Activities"].get(
            "Acquisitions (net of cash acquired)", []
        )

        has_simple_version = any(
            "payments to acquire businesses net of cash acquired" in a.lower()
            and len(a) < len(feature)
            for a in acquisitions
        )

        if has_simple_version:
            short = feature[:60]
            msg = f"✓ Verified duplicate acquisition feature unmapped: '{short}...'"
            print(msg)
        else:
            unmapped.remove(feature)
            acquisitions.append(feature)
            print(f"✓ Post-processing fix: '{feature[:60]}...' → Acquisitions")


def _fix_cash_flow_acquisitions(mapped_structure: dict) -> dict:
    """Fix acquisition-related cash flow items."""
    unmapped = mapped_structure.get("__unmapped__", [])
    _fix_acquisitions_net_of_cash(mapped_structure, unmapped)
    _fix_net_change_in_cash(mapped_structure, unmapped)
    _fix_payments_to_acquire(mapped_structure, unmapped)
    return mapped_structure


def _fix_cash_flow_operating(mapped_structure: dict) -> dict:
    """Fix operating activities cash flow items."""
    unmapped = mapped_structure.get("__unmapped__", [])

    for feature in list(unmapped):
        if (
            "other operating activities" in feature.lower()
            and "cash flow" in feature.lower()
        ):
            unmapped.remove(feature)
            if "Operating Activities" in mapped_structure:
                other_noncash = mapped_structure["Operating Activities"].get(
                    "Other Non-Cash Items", []
                )
                other_noncash.append(feature)
                print(f"✓ Post-processing fix: '{feature}' → Other Non-Cash Items")

    return mapped_structure


def _fix_cash_flow_debt_proceeds(mapped_structure: dict) -> dict:
    """Fix debt proceeds classification in cash flow."""
    if not (
        "Financing Activities" in mapped_structure
        and "Investing Activities" in mapped_structure
    ):
        return mapped_structure

    proceeds_debt = mapped_structure["Financing Activities"].get(
        "Proceeds from Debt", []
    )
    investment_sales = mapped_structure["Investing Activities"].get(
        "Sales and Maturities of Investments", []
    )

    for feature in list(proceeds_debt):
        if "sale of" in feature.lower() and "securities" in feature.lower():
            proceeds_debt.remove(feature)
            if not isinstance(investment_sales, list):
                mapped_structure["Investing Activities"][
                    "Sales and Maturities of Investments"
                ] = []
                investment_sales = mapped_structure["Investing Activities"][
                    "Sales and Maturities of Investments"
                ]
            investment_sales.append(feature)
            short = feature[:50]
            msg = f"✓ Post-processing fix: '{short}...' moved from "
            msg += "Proceeds from Debt → Sales and Maturities of Investments"
            print(msg)

    categories_to_check = [
        ("Financing Activities", "Repayment of Debt"),
        ("Investing Activities", "Sales and Maturities of Investments"),
        ("Investing Activities", "Other Investing Activities"),
    ]

    for section, category in categories_to_check:
        if section in mapped_structure:
            category_list = mapped_structure[section].get(category, [])
            for feature in list(category_list):
                if (
                    feature.lower().startswith("proceeds from")
                    and "debt" in feature.lower()
                ):
                    category_list.remove(feature)
                    if not isinstance(proceeds_debt, list):
                        mapped_structure["Financing Activities"][
                            "Proceeds from Debt"
                        ] = []
                        proceeds_debt = mapped_structure["Financing Activities"][
                            "Proceeds from Debt"
                        ]
                    proceeds_debt.append(feature)
                    short = feature[:50]
                    msg = f"✓ Post-processing fix: '{short}...' moved from "
                    msg += f"{category} → Proceeds from Debt"
                    print(msg)

    return mapped_structure


def _fix_cash_flow_noncurrent_assets(mapped_structure: dict) -> dict:
    """Fix noncurrent assets classification in cash flow."""
    if not (
        "Operating Activities" in mapped_structure
        and "Investing Activities" in mapped_structure
    ):
        return mapped_structure

    operating_changes = mapped_structure["Operating Activities"].get(
        "Changes in Operating Assets and Liabilities", []
    )
    other_investing = mapped_structure["Investing Activities"].get(
        "Other Investing Activities", []
    )
    unmapped = mapped_structure.get("__unmapped__", [])

    for feature in list(other_investing) + list(unmapped):
        if "increase decrease in other noncurrent assets" in feature.lower():
            if feature in other_investing:
                other_investing.remove(feature)
            if feature in unmapped:
                unmapped.remove(feature)

            if feature not in operating_changes:
                operating_changes.append(feature)
                msg = f"✓ Post-processing fix: '{feature}' moved from "
                msg += "Other Investing → Operating Changes"
                print(msg)

    return mapped_structure


def _fix_cash_flow(mapped_structure: dict) -> dict:
    """Apply fixes for cash flow statement edge cases."""
    mapped_structure = _fix_cash_flow_acquisitions(mapped_structure)
    mapped_structure = _fix_cash_flow_operating(mapped_structure)
    mapped_structure = _fix_cash_flow_debt_proceeds(mapped_structure)
    mapped_structure = _fix_cash_flow_noncurrent_assets(mapped_structure)
    return mapped_structure


def apply_statement_specific_fixes(
    statement_type: str, mapped_structure: dict, features: List[str]
) -> dict:
    """Apply statement-specific post-processing fixes for known edge cases."""
    if statement_type == "income_statement":
        return _fix_income_statement(mapped_structure)
    elif statement_type == "balance_sheet":
        return _fix_balance_sheet(mapped_structure)
    elif statement_type == "cash_flow":
        return _fix_cash_flow(mapped_structure)
    return mapped_structure


FX_RATES = {
    ("EUR", "2024-12-31"): 1.0350501858534082,
    ("EUR", "2023-12-31"): 1.103896762359319,
    ("USD", "2024-12-31"): 1.0000,
    ("USD", "2023-12-31"): 1.0000,
    ("JPY", "2024-12-31"): 0.006353549227213416,
    ("JPY", "2023-12-31"): 0.0070924872570355384,
    ("CNY", "2024-12-31"): 0.1369969389554929,
    ("CNY", "2023-12-31"): 0.1409019797593253,
    ("GBP", "2024-12-31"): 1.2520856867720418,
    ("GBP", "2023-12-31"): 1.2732493756832426,
}


def get_fx_rate(currency: str, date: str) -> float:
    """Get historical FX rate from API or fallback to hardcoded."""
    if currency == "USD":
        return 1.0
    try:
        import requests

        fx_api = os.getenv("FX_API_KEY")
        if not fx_api:
            raise ValueError(
                "FX_API_KEY environment variable not set"
                "set with 'export FX_API_KEY='your_api_key'"
            )

        stripped_date = date.strip().split("-")
        url = (
            f"https://v6.exchangerate-api.com/v6/{fx_api}/history/USD/"
            f"{stripped_date[0]}/{stripped_date[1]}/{stripped_date[2]}"
        )

        response = requests.get(url)
        data = response.json()

        fx_rate = 1 / data["conversion_rates"][currency]
        print(f"Using exchange rate: {fx_rate} for {currency} on {date}")
        return fx_rate
    except Exception:
        print(f"Warning: Using fallback FX rate for {currency} on {date}")
        return FX_RATES.get((currency, date), 1.0)
