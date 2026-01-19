"""Tests for clients/utils.py - LLM response parsing and financial statement fixes."""

from unittest.mock import patch

import pytest

from jpm.question_1.clients import utils

unit = pytest.mark.unit
integration = pytest.mark.integration


# Tests for count_leaf_list_values and leaf_list_values


@unit
def test_count_leaf_list_values_flat():
    """count_leaf_list_values should count items in flat dict."""
    d = {
        "key1": ["a", "b", "c"],
        "key2": ["d"],
    }
    assert utils.count_leaf_list_values(d) == 4


@unit
def test_count_leaf_list_values_nested():
    """count_leaf_list_values should count items in nested dict."""
    d = {
        "outer": {
            "inner1": ["a", "b"],
            "inner2": {"deep": ["c", "d", "e"]},
        },
        "flat": ["f"],
    }
    assert utils.count_leaf_list_values(d) == 6


@unit
def test_count_leaf_list_values_empty():
    """count_leaf_list_values should return 0 for empty dict."""
    assert utils.count_leaf_list_values({}) == 0


@unit
def test_leaf_list_values_flat():
    """leaf_list_values should return all leaf values from flat dict."""
    d = {"key1": ["a", "b"], "key2": ["c"]}
    result = utils.leaf_list_values(d)
    assert sorted(result) == ["a", "b", "c"]


@unit
def test_leaf_list_values_nested():
    """leaf_list_values should return all leaf values from nested dict."""
    d = {
        "outer": {
            "inner": ["a", "b"],
        },
        "flat": ["c"],
    }
    result = utils.leaf_list_values(d)
    assert sorted(result) == ["a", "b", "c"]


# Tests for income statement fix functions


@unit
def test_is_income_before_taxes_feature():
    """_is_income_before_taxes_feature should detect IBT features."""
    assert utils._is_income_before_taxes_feature("Income before income taxes") is True
    assert utils._is_income_before_taxes_feature("Pretax income") is True
    assert utils._is_income_before_taxes_feature("Income Before Taxes") is True
    assert utils._is_income_before_taxes_feature("Net Income") is False
    assert utils._is_income_before_taxes_feature("Revenue") is False


@unit
def test_fix_income_before_taxes_moves_from_unmapped(capsys):
    """_fix_income_before_taxes should move IBT features to correct location."""
    mapped = {"Revenue": ["rev_feature"]}
    unmapped = ["Income before income taxes", "other_feature"]

    utils._fix_income_before_taxes(mapped, unmapped)

    assert "Income before income taxes" not in unmapped
    assert "Income Before Taxes" in mapped
    assert "Income before income taxes" in mapped["Income Before Taxes"]
    assert "other_feature" in unmapped


@unit
def test_dedupe_income_before_taxes_keeps_shortest(capsys):
    """_dedupe_income_before_taxes should keep shortest and move others to unmapped."""
    mapped = {
        "Income Before Taxes": [
            "ShortName",
            "A Much Longer Income Before Taxes Name",
        ],
        "__unmapped__": [],
    }

    utils._dedupe_income_before_taxes(mapped)

    assert mapped["Income Before Taxes"] == ["ShortName"]
    assert "A Much Longer Income Before Taxes Name" in mapped["__unmapped__"]


@unit
def test_fix_income_statement_applies_all_fixes():
    """_fix_income_statement should apply IBT and non-op fixes."""
    mapped = {
        "Revenue": ["rev"],
        "Non-Operating Items": {"Total Non-Operating Items": []},
        "__unmapped__": [
            "income before taxes feature",
            "nonoperating income expense",
        ],
    }

    result = utils._fix_income_statement(mapped)

    # IBT should be extracted
    assert "Income Before Taxes" in result
    # Non-op should be moved
    assert (
        "nonoperating income expense"
        in result["Non-Operating Items"]["Total Non-Operating Items"]
    )


# Tests for balance sheet fix functions


@unit
def test_fix_balance_sheet_verifies_share_counts(capsys):
    """_fix_balance_sheet should verify unmapped share count features."""
    mapped = {
        "Assets": ["asset1"],
        "__unmapped__": ["common shares issued", "other_unmapped"],
    }

    utils._fix_balance_sheet(mapped)

    captured = capsys.readouterr()
    assert "Verified unmapped share count" in captured.out


# Tests for cash flow fix functions


@unit
def test_is_net_change_in_cash():
    """_is_net_change_in_cash should detect net cash change features."""
    assert (
        utils._is_net_change_in_cash(
            "Cash and cash equivalents period increase decrease"
        )
        is True
    )
    assert utils._is_net_change_in_cash("Cash equivalents increase decrease") is True
    assert utils._is_net_change_in_cash("Revenue") is False


@unit
def test_fix_net_change_in_cash_moves_feature(capsys):
    """_fix_net_change_in_cash should move matching features."""
    mapped = {"Operating Activities": {}}
    unmapped = ["cash and cash equivalents period increase decrease"]

    utils._fix_net_change_in_cash(mapped, unmapped)

    assert "Net Change in Cash" in mapped
    assert len(unmapped) == 0


@unit
def test_fix_acquisitions_net_of_cash(capsys):
    """_fix_acquisitions_net_of_cash should categorize acquisition features."""
    mapped = {
        "Investing Activities": {"Acquisitions (net of cash acquired)": []},
        "__unmapped__": ["payments for acquisitions net of cash acquired"],
    }
    unmapped = mapped["__unmapped__"]

    utils._fix_acquisitions_net_of_cash(mapped, unmapped)

    assert len(unmapped) == 0
    assert (
        "payments for acquisitions net of cash acquired"
        in mapped["Investing Activities"]["Acquisitions (net of cash acquired)"]
    )


@unit
def test_apply_statement_specific_fixes_income_statement():
    """apply_statement_specific_fixes should route to income statement fixes."""
    mapped = {"__unmapped__": []}

    result = utils.apply_statement_specific_fixes(
        "income_statement", mapped, features=[]
    )

    assert result is not None


@unit
def test_apply_statement_specific_fixes_balance_sheet():
    """apply_statement_specific_fixes should route to balance sheet fixes."""
    mapped = {"__unmapped__": []}

    result = utils.apply_statement_specific_fixes("balance_sheet", mapped, features=[])

    assert result is not None


@unit
def test_apply_statement_specific_fixes_cash_flow():
    """apply_statement_specific_fixes should route to cash flow fixes."""
    mapped = {"__unmapped__": []}

    result = utils.apply_statement_specific_fixes("cash_flow", mapped, features=[])

    assert result is not None


@unit
def test_apply_statement_specific_fixes_unknown_type():
    """apply_statement_specific_fixes should return unchanged for unknown type."""
    mapped = {"key": "value"}

    result = utils.apply_statement_specific_fixes("unknown", mapped, features=[])

    assert result == mapped


# Tests for FX rate functions


@unit
def test_get_fx_rate_usd_returns_one():
    """get_fx_rate should return 1.0 for USD."""
    assert utils.get_fx_rate("USD", "2024-01-01") == 1.0


@unit
def test_get_fx_rate_fallback_rates():
    """get_fx_rate should use fallback rates when API unavailable."""
    # Without FX_API_KEY set, should use fallback
    with patch.dict("os.environ", {}, clear=True):
        rate = utils.get_fx_rate("EUR", "2024-12-31")
        assert rate == pytest.approx(1.0350501858534082)

        rate = utils.get_fx_rate("JPY", "2023-12-31")
        assert rate == pytest.approx(0.0070924872570355384)


@unit
def test_get_fx_rate_unknown_currency_fallback():
    """get_fx_rate should return 1.0 for unknown currency/date combo."""
    with patch.dict("os.environ", {}, clear=True):
        rate = utils.get_fx_rate("XYZ", "2020-01-01")
        assert rate == 1.0


# Tests for JSON parsing functions


@unit
def test_parse_llm_json_response_clean_json():
    """parse_llm_json_response should parse clean JSON."""
    response = '{"key": "value", "number": 42}'
    result = utils.parse_llm_json_response(response)

    assert result == {"key": "value", "number": 42}


@unit
def test_parse_llm_json_response_with_markdown():
    """parse_llm_json_response should handle markdown code blocks."""
    response = """```json
{"key": "value"}
```"""
    result = utils.parse_llm_json_response(response)

    assert result == {"key": "value"}


@unit
def test_parse_llm_json_response_with_surrounding_text():
    """parse_llm_json_response should extract JSON from surrounding text."""
    response = """Here is the result:
{"key": "value"}
Hope this helps!"""
    result = utils.parse_llm_json_response(response)

    assert result == {"key": "value"}


@unit
def test_parse_llm_json_response_nested():
    """parse_llm_json_response should handle nested structures."""
    response = '{"outer": {"inner": [1, 2, 3]}, "flag": true}'
    result = utils.parse_llm_json_response(response)

    assert result == {"outer": {"inner": [1, 2, 3]}, "flag": True}


@unit
def test_parse_llm_json_response_invalid_raises():
    """parse_llm_json_response should raise ValueError for invalid JSON."""
    response = "This is not valid JSON at all"

    with pytest.raises(ValueError, match="Could not parse JSON"):
        utils.parse_llm_json_response(response)


@unit
def test_parse_llm_list_response_clean_list():
    """parse_llm_list_response should parse clean JSON list."""
    response = '["item1", "item2", "item3"]'
    result = utils.parse_llm_list_response(response)

    assert result == ["item1", "item2", "item3"]


@unit
def test_parse_llm_list_response_with_markdown():
    """parse_llm_list_response should handle markdown code blocks."""
    response = """```json
["a", "b", "c"]
```"""
    result = utils.parse_llm_list_response(response)

    assert result == ["a", "b", "c"]


@unit
def test_parse_llm_list_response_with_surrounding_text():
    """parse_llm_list_response should extract list from surrounding text."""
    response = 'The items are: ["one", "two"] as requested.'
    result = utils.parse_llm_list_response(response)

    assert result == ["one", "two"]


@unit
def test_parse_llm_list_response_filters_none():
    """parse_llm_list_response should filter out None values."""
    response = '["a", null, "b", null]'
    result = utils.parse_llm_list_response(response)

    assert result == ["a", "b"]


@unit
def test_parse_llm_list_response_converts_to_strings():
    """parse_llm_list_response should convert items to strings."""
    response = '[1, 2.5, true, "text"]'
    result = utils.parse_llm_list_response(response)

    assert result == ["1", "2.5", "True", "text"]


@unit
def test_parse_llm_list_response_dict_raises():
    """parse_llm_list_response should raise ValueError for dict input."""
    response = '{"key": "value"}'

    with pytest.raises(ValueError, match="Expected list"):
        utils.parse_llm_list_response(response)


@unit
def test_parse_llm_list_response_invalid_raises():
    """parse_llm_list_response should raise ValueError for invalid JSON."""
    response = "Not a valid list"

    with pytest.raises(ValueError, match="Could not parse JSON list"):
        utils.parse_llm_list_response(response)


# Tests for cash flow debt proceeds fixes


@unit
def test_fix_cash_flow_debt_proceeds_moves_securities():
    """_fix_cash_flow_debt_proceeds should move security sales to investing."""
    mapped = {
        "Financing Activities": {
            "Proceeds from Debt": ["proceeds from sale of securities investment"],
        },
        "Investing Activities": {
            "Sales and Maturities of Investments": [],
        },
    }

    utils._fix_cash_flow_debt_proceeds(mapped)

    assert (
        "proceeds from sale of securities investment"
        not in mapped["Financing Activities"]["Proceeds from Debt"]
    )
    assert (
        "proceeds from sale of securities investment"
        in mapped["Investing Activities"]["Sales and Maturities of Investments"]
    )


@unit
def test_fix_cash_flow_noncurrent_assets_moves_to_operating():
    """_fix_cash_flow_noncurrent_assets should move noncurrent asset changes."""
    mapped = {
        "Operating Activities": {
            "Changes in Operating Assets and Liabilities": [],
        },
        "Investing Activities": {
            "Other Investing Activities": [
                "increase decrease in other noncurrent assets"
            ],
        },
        "__unmapped__": [],
    }

    utils._fix_cash_flow_noncurrent_assets(mapped)

    assert (
        "increase decrease in other noncurrent assets"
        in mapped["Operating Activities"]["Changes in Operating Assets and Liabilities"]
    )


# Edge cases and error handling


@unit
def test_fix_nonop_duplicates_handles_missing_section():
    """_fix_nonop_duplicates should handle missing Non-Operating Items."""
    mapped = {"Revenue": ["rev"]}

    # Should not raise
    utils._fix_nonop_duplicates(mapped)


@unit
def test_fix_cash_flow_operating_moves_other_operating():
    """_fix_cash_flow_operating should move other operating activities."""
    mapped = {
        "Operating Activities": {
            "Other Non-Cash Items": [],
        },
        "__unmapped__": ["other operating activities cash flow statement"],
    }

    utils._fix_cash_flow_operating(mapped)

    assert (
        "other operating activities cash flow statement" not in mapped["__unmapped__"]
    )
    assert (
        "other operating activities cash flow statement"
        in mapped["Operating Activities"]["Other Non-Cash Items"]
    )
