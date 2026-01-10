from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from io import BytesIO, StringIO
from typing import Any, Dict, List, Optional

import pandas as pd
from jpm.question_1.config import LLMConfig
from jpm.question_1.data.structures import get_fs_struct
from jpm.question_1.misc import format_money
from openai import OpenAI
from pandas import DataFrame
from pypdf import PdfReader, PdfWriter


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


def apply_statement_specific_fixes(
    statement_type: str, mapped_structure: dict, features: List[str]
) -> dict:
    """Apply statement-specific post-processing fixes for known edge cases."""

    if statement_type == "income_statement":
        unmapped = mapped_structure.get("__unmapped__", [])

        # Fix: Income Before Taxes
        for feature in list(unmapped):
            if (
                "before income taxes" in feature.lower()
                or "before taxes" in feature.lower()
                or "pretax" in feature.lower()
            ):
                unmapped.remove(feature)
                if "Income Before Taxes" not in mapped_structure:
                    mapped_structure["Income Before Taxes"] = []
                if not isinstance(mapped_structure.get("Income Before Taxes"), list):
                    mapped_structure["Income Before Taxes"] = []
                mapped_structure["Income Before Taxes"].append(feature)
                print(
                    f"✓ Post-processing fix: '{feature[:60]}...' → Income Before Taxes"
                )

        # Fix: Deduplicate Income Before Taxes if multiple similar features exist
        ibt_features = mapped_structure.get("Income Before Taxes", [])
        if len(ibt_features) > 1:
            # Keep the shorter/simpler name (usually more standard)
            ibt_features_sorted = sorted(ibt_features, key=len)
            kept_feature = ibt_features_sorted[0]
            removed_features = ibt_features_sorted[1:]

            mapped_structure["Income Before Taxes"] = [kept_feature]
            mapped_structure.setdefault("__unmapped__", []).extend(removed_features)

            print("⚠ Warning: Multiple 'Income Before Taxes' features detected")
            print(f"  Kept: '{kept_feature}'")
            for rf in removed_features:
                print(f"  Moved to unmapped: '{rf}'")

        # Fix: Total Non-Operating Items
        for feature in list(unmapped):
            if feature.lower() == "nonoperating income expense":
                unmapped.remove(feature)
                if "Non-Operating Items" in mapped_structure:
                    if not isinstance(
                        mapped_structure["Non-Operating Items"].get(
                            "Total Non-Operating Items"
                        ),
                        list,
                    ):
                        mapped_structure["Non-Operating Items"][
                            "Total Non-Operating Items"
                        ] = []
                    mapped_structure["Non-Operating Items"][
                        "Total Non-Operating Items"
                    ].append(feature)
                    print(
                        f"✓ Post-processing fix: '{feature}' → Total Non-Operating Items"
                    )

        # Fix: Check for duplicates in Total Non-Operating Items
        if "Non-Operating Items" in mapped_structure:
            total_nonop = mapped_structure["Non-Operating Items"].get(
                "Total Non-Operating Items", []
            )
            other_nonop = mapped_structure["Non-Operating Items"].get(
                "Other Income (Expense)", []
            )

            if len(total_nonop) > 1:
                # Check if one contains "other" - it's likely a component, not total
                for feature in list(total_nonop):
                    if "other nonoperating" in feature.lower() and len(total_nonop) > 1:
                        total_nonop.remove(feature)
                        other_nonop.append(feature)
                        print(
                            f"✓ Post-processing fix: '{feature}' moved from Total Non-Op → Other Income (Expense)"
                        )

    elif statement_type == "balance_sheet":
        unmapped = mapped_structure.get("__unmapped__", [])

        # Fix: Share counts that shouldn't be in financial categories
        for feature in list(unmapped):
            # This is actually correct - share counts should stay unmapped
            # But we can add logging for verification
            if "shares" in feature.lower() and (
                "issued" in feature.lower() or "outstanding" in feature.lower()
            ):
                print(f"✓ Verified unmapped share count: '{feature}'")

    elif statement_type == "cash_flow":
        unmapped = mapped_structure.get("__unmapped__", [])

        # Fix: Acquisitions
        for feature in list(unmapped):
            if "acquisitions net of cash acquired" in feature.lower():
                unmapped.remove(feature)
                if "Investing Activities" in mapped_structure:
                    if not isinstance(
                        mapped_structure["Investing Activities"].get(
                            "Acquisitions (net of cash acquired)"
                        ),
                        list,
                    ):
                        mapped_structure["Investing Activities"][
                            "Acquisitions (net of cash acquired)"
                        ] = []
                    mapped_structure["Investing Activities"][
                        "Acquisitions (net of cash acquired)"
                    ].append(feature)
                    print(f"✓ Post-processing fix: '{feature[:60]}...' → Acquisitions")

        # Fix: Net Change in Cash
        for feature in list(unmapped):
            if (
                "cash" in feature.lower()
                and "period increase decrease" in feature.lower()
            ) or (
                "cash equivalents" in feature.lower()
                and "increase decrease" in feature.lower()
            ):
                unmapped.remove(feature)
                if not isinstance(mapped_structure.get("Net Change in Cash"), list):
                    mapped_structure["Net Change in Cash"] = []
                mapped_structure["Net Change in Cash"].append(feature)
                print(
                    f"✓ Post-processing fix: '{feature[:60]}...' → Net Change in Cash"
                )

        # Fix: Check for duplicate acquisition features
        for feature in list(unmapped):
            if "payments to acquire businesses" in feature.lower():
                # Check if simpler version is already mapped
                acquisitions = mapped_structure["Investing Activities"].get(
                    "Acquisitions (net of cash acquired)", []
                )

                has_simple_version = any(
                    "payments to acquire businesses net of cash acquired" in a.lower()
                    and len(a) < len(feature)
                    for a in acquisitions
                )

                if has_simple_version:
                    print(
                        f"✓ Verified duplicate acquisition feature unmapped: '{feature[:60]}...'"
                    )
                else:
                    # No duplicate, this should be mapped
                    unmapped.remove(feature)
                    acquisitions.append(feature)
                    print(f"✓ Post-processing fix: '{feature[:60]}...' → Acquisitions")

        # Fix: "other operating activities cash flow statement" should be in Other Non-Cash Items
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

        # Fix: Debt features that start with "proceeds from" should be in Proceeds from Debt
        # Check ALL categories where it might have been misplaced
        if (
            "Financing Activities" in mapped_structure
            and "Investing Activities" in mapped_structure
        ):
            proceeds_debt = mapped_structure["Financing Activities"].get(
                "Proceeds from Debt", []
            )
            investment_sales = mapped_structure["Investing Activities"].get(
                "Sales and Maturities of Investments", []
            )

            # Check for investment securities sales misclassified as debt proceeds
            for feature in list(proceeds_debt):
                # "sale of available for sale securities" = selling investments, not issuing debt
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
                    print(
                        f"✓ Post-processing fix: '{feature[:50]}...' moved from Proceeds from Debt → Sales and Maturities of Investments"
                    )

            # Categories to check for misplaced debt proceeds
            categories_to_check = [
                ("Financing Activities", "Repayment of Debt"),
                ("Investing Activities", "Sales and Maturities of Investments"),
                ("Investing Activities", "Other Investing Activities"),
            ]

            for section, category in categories_to_check:
                if section in mapped_structure:
                    category_list = mapped_structure[section].get(category, [])
                    for feature in list(category_list):
                        # Check if it's a debt-related feature that starts with "proceeds from"
                        if (
                            feature.lower().startswith("proceeds from")
                            and "debt" in feature.lower()
                        ):
                            category_list.remove(feature)
                            if not isinstance(proceeds_debt, list):
                                mapped_structure["Financing Activities"][
                                    "Proceeds from Debt"
                                ] = []
                                proceeds_debt = mapped_structure[
                                    "Financing Activities"
                                ]["Proceeds from Debt"]
                            proceeds_debt.append(feature)
                            print(
                                f"✓ Post-processing fix: '{feature[:50]}...' moved from {category} → Proceeds from Debt"
                            )

        # Fix: "increase decrease in other noncurrent assets" is operating, not investing
        if (
            "Operating Activities" in mapped_structure
            and "Investing Activities" in mapped_structure
        ):
            operating_changes = mapped_structure["Operating Activities"].get(
                "Changes in Operating Assets and Liabilities", []
            )
            other_investing = mapped_structure["Investing Activities"].get(
                "Other Investing Activities", []
            )
            unmapped = mapped_structure.get("__unmapped__", [])

            # Check both Other Investing and unmapped
            for feature in list(other_investing) + list(unmapped):
                if "increase decrease in other noncurrent assets" in feature.lower():
                    if feature in other_investing:
                        other_investing.remove(feature)
                    if feature in unmapped:
                        unmapped.remove(feature)

                    if feature not in operating_changes:
                        operating_changes.append(feature)
                        print(
                            f"✓ Post-processing fix: '{feature}' moved from Other Investing → Operating Changes"
                        )

    return mapped_structure


class LLMClient:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
    ) -> None:
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            raise RuntimeError(
                "OpenAI API key is missing. "
                "Set OPENAI_API_KEY or pass openai_api_key explicitly."
            )

        self._openai: OpenAI = OpenAI(api_key=openai_api_key)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        cfg: LLMConfig,
    ) -> str:
        if cfg.provider == "openai":
            return self._chat_openai(messages, cfg)
        else:
            raise ValueError(f"Unknown provider: {cfg.provider}")

    def _chat_openai(
        self,
        messages: List[Dict[str, Any]],
        cfg: LLMConfig,
    ) -> str:
        if self._openai is None:
            raise RuntimeError("OpenAI client not initialised (missing API key).")

        with self._spinner(f"Waiting for {cfg.provider} {cfg.model} response"):
            # Tune tokens
            if "nano" not in cfg.model:
                resp = self._openai.chat.completions.create(
                    model=cfg.model,
                    messages=messages,
                    # temperature=cfg.temperature,
                    # max_completion_tokens=cfg.max_tokens,
                )
            else:
                resp = self._openai.chat.completions.create(
                    model=cfg.model,
                    messages=messages,
                    # max_completion_tokens=cfg.max_tokens,
                )
        return resp.choices[0].message.content

    @contextmanager
    def _spinner(self, message: str, delay: float = 0.1):
        """Lightweight terminal spinner with elapsed timer for API calls."""
        stop = threading.Event()
        start = time.time()
        colors = ["\033[91m", "\033[93m", "\033[92m", "\033[94m", "\033[95m"]
        reset = "\033[0m"

        def _spin():
            symbols = "|/-\\"
            i = 0
            while not stop.is_set():
                elapsed = time.time() - start
                color = colors[i % len(colors)]
                sys.stdout.write(
                    f"\r{color}{message} {symbols[i % len(symbols)]}"
                    f"{elapsed:>5.1f}s{reset}"
                )
                sys.stdout.flush()
                time.sleep(delay)
                i += 1
            total = time.time() - start
            sys.stdout.write(f"\r{message} done in {total:.2f}s.\n")
            sys.stdout.flush()

        thread = threading.Thread(target=_spin, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()

    def forecast_next_quarter(
        self,
        history: DataFrame,
        cfg: LLMConfig,
        prediction: DataFrame | None = None,
        feature_columns: Optional[List[str]] = None,
        max_rows: int = 4,
        adjust: bool = False,
    ) -> DataFrame:
        if history is None or history.empty:
            raise ValueError("DataFrame must contain quarterly financial data.")

        target_cols = feature_columns or list(history.columns)
        missing = [col for col in target_cols if col not in history.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

        recent = history.sort_index().tail(max_rows)[target_cols]
        data_csv = recent.to_csv(index=True)

        if adjust:
            assert prediction is not None, (
                "Prediction DataFrame required for adjust=True"
            )
            predict_csv = prediction.to_csv(index=True)
            system_content = (
                "You are a financial analyst. "
                "The user will provide a time-ordered DataFrame of quarterly "
                "financial statement metrics. "
                "The final row is an estimation of the next quarters financials."
                "Improve the prediction for every provided feature, using deep"
                "reasoning and knowledge of recent market trends etc."
            )
            user_content = (
                "Here is the quarterly time series (oldest to newest):\n"
                f"{data_csv}\n"
                "And here is the prediction for the next quarter:\n"
                f"{predict_csv}\n"
                "Return ONLY valid CSV with header row matching these columns and "
                "a single row labeled with the next quarter's YYYY-MM-DD, containing "
                "the next quarter's improved estimated values for each column."
            )
        else:
            system_content = (
                "You are a financial analyst. "
                "The user will provide a time-ordered DataFrame of quarterly "
                "financial statement metrics. Estimate the next quarter's values "
                "for every provided feature."
            )
            user_content = (
                "Here is the quarterly time series (oldest to newest):\n"
                f"{data_csv}\n"
                "Return ONLY valid CSV with header row matching these columns and "
                "a single row labeled with the next quarter's YYYY-MM-DD, containing "
                "the next quarter's estimated values for each column."
            )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        raw = self.chat(messages, cfg)

        try:
            parsed = pd.read_csv(StringIO(raw), index_col=0)
        except Exception as exc:  # pragma: no cover - defensive parsing
            raise ValueError(f"Failed to parse LLM response as CSV: {exc}") from exc

        missing_cols = [c for c in target_cols if c not in parsed.columns]

        if missing_cols:
            raise ValueError(f"LLM response missing columns: {missing_cols}")

        return parsed[target_cols].head(1)

    def parse_financial_features(
        self,
        features: List[str],
        statement_type: str,
        cfg: LLMConfig,
    ) -> Dict[str, Any]:
        """
        Map raw EDGAR feature strings to standardized financial statement structure.

        Args:
            statement_type: One of 'balance_sheet', 'income_statement', 'cash_flow'
            features: List of raw column names from EDGAR
            cfg: Configuration dict for API call

        Returns:
            Dictionary with features mapped to prediction_structure
        """
        # Get the appropriate statement structure
        fs_dictionary = get_fs_struct(statement_type)

        structure_json = fs_dictionary["prediction_structure"]
        mapping_examples = fs_dictionary["mapping_examples"]
        guidelines = fs_dictionary["classification_guidelines"]

        structure_str = json.dumps(
            structure_json, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        features_str = json.dumps(features, ensure_ascii=False)

        # Build context strings
        examples_context = "\n".join(
            [
                f"- {category}: {', '.join(examples[:5])}"
                for category, examples in mapping_examples.items()
            ]
        )

        general_rules = "\n".join(
            [f"  - {rule}" for rule in guidelines["General Rules"]]
        )
        specific_mappings = "\n".join(
            [
                f"  - {category}: {desc}"
                for category, desc in guidelines["Specific Mappings"].items()
            ]
        )
        edge_cases = "\n".join(
            [
                f"  - {case}: {desc}"
                for case, desc in guidelines.get("Edge Cases", {}).items()
            ]
        )

        # Add rollup detection if present
        rollup_detection = ""
        if "Rollup Detection" in guidelines:
            rollup_rules = guidelines["Rollup Detection"]
            if isinstance(rollup_rules, dict):
                lines = []
                for category, rules in rollup_rules.items():
                    lines.append(f"  {category}:")
                    for rule in rules:
                        lines.append(f"    - {rule}")
                rollup_detection = "\n".join(lines)
            elif isinstance(rollup_rules, list):
                rollup_detection = "\n".join([f"  - {rule}" for rule in rollup_rules])

        # Build statement-specific keyword rules
        keyword_rules = ""
        if statement_type == "income_statement":
            keyword_rules = """
                CRITICAL KEYWORD MATCHING RULES FOR INCOME STATEMENT (APPLY WITH CONFIDENCE):
                If a feature contains these keywords, map it regardless of verbose phrasing:

                1. "before income taxes" OR "before taxes" OR "pretax" → Income Before Taxes
                - Example: "income loss from continuing operations before income taxes..." → Income Before Taxes
                2. "nonoperating income expense" (exact or close match) → Total Non-Operating Items
                3. "costs and expenses" OR "total costs and expenses" → __unmapped__ (rollup of COGS + OpEx)
                4. "fulfillment" → Other Operating Expenses (NOT Cost of Revenue)
                5. "technology and" (content/infrastructure/development) → Research and Development

                These patterns override general caution - apply them confidently.
                """
        elif statement_type == "balance_sheet":
            keyword_rules = """
                CRITICAL KEYWORD MATCHING RULES FOR BALANCE SHEET (APPLY WITH CONFIDENCE):
                If a feature contains these keywords, map it regardless of verbose phrasing:

                1. Exact "assets" (not "assets current" or "assets noncurrent") → Total Assets
                2. Exact "liabilities" (not "liabilities current" or "liabilities noncurrent") → Total Liabilities
                3. "stockholders equity" OR "shareholders equity" → Total Equity
                4. "shares issued" OR "shares outstanding" OR "shares authorized" → __unmapped__ (share counts, not dollars)
                5. Features combining multiple concepts (e.g., "cash cash equivalents and short term investments") → Check if components exist separately, if yes → __unmapped__ (rollup)

                These patterns override general caution - apply them confidently.
                """
        elif statement_type == "cash_flow":
            keyword_rules = """
                CRITICAL KEYWORD MATCHING RULES FOR CASH FLOW (APPLY WITH CONFIDENCE):
                If a feature contains these keywords, map it regardless of verbose phrasing:

                1. "net cash provided by operating" OR "net cash from operating" → Net Cash from Operating Activities
                2. "net cash used in investing" OR "net cash from investing" → Net Cash from Investing Activities
                3. "net cash provided by financing" OR "net cash from financing" → Net Cash from Financing Activities
                4. "changes in working capital" OR "changes in operating assets and liabilities" → Changes in Operating Assets and Liabilities
                5. "acquisitions net of cash acquired" (with or without additional text like "and purchases of...") → Acquisitions (net of cash acquired)
                6. "cash" AND "period increase decrease" → Net Change in Cash
                7. "cash equivalents" AND "increase decrease including exchange rate" → Net Change in Cash

                These patterns override general caution and rollup detection - apply them confidently.
                """

        system_content = f"""
            You are a deterministic mapper from feature strings to a fixed {statement_type.replace("_", " ")} JSON taxonomy.

            DEFINITIONS:
            - A "leaf" is any key whose value is a JSON array (list). Only leaves may be modified.
            - Non-leaves are JSON objects (dicts) and must not be modified.

            HARD REQUIREMENTS:
            - Output must be a single valid JSON object.
            - The output object MUST keep identical keys and nesting as the input structure (no new keys; no renamed keys).
            - Only append items to existing leaf arrays.
            - Feature strings must be copied verbatim (exact characters; no normalization).

            {keyword_rules}

            CRITICAL TOTAL/ROLLUP MAPPING RULES (APPLY FIRST):
            Balance Sheet totals MUST be mapped as follows:
            1. Features matching "assets" (exact) or "total assets" → Assets / Total Assets
            2. Features matching "assets current" or "total current assets" → Assets / Current / Total Current Assets
            3. Features matching "assets noncurrent" or "total non-current assets" or "assets non current" → Assets / Non-Current / Total Non-Current Assets
            4. Features matching "liabilities" (exact) or "total liabilities" → Liabilities / Total Liabilities
            5. Features matching "liabilities current" or "total current liabilities" → Liabilities / Current / Total Current Liabilities
            6. Features matching "liabilities noncurrent" or "total non-current liabilities" → Liabilities / Non-Current / Total Non-Current Liabilities
            7. Features matching "stockholders equity" or "shareholders equity" or "total equity" → Equity / Total Equity
            8. Features matching "liabilities and stockholders equity" or similar → Totals / Total Liabilities and Equity

            Income Statement totals:
            1. Features matching "revenue" or "total revenue" or "net sales" → Revenues / Total Revenues
            2. Features matching "cost of revenue" or "cost of sales" → Cost of Revenue / Total Cost of Revenue
            3. Features matching "operating expenses" → Operating Expenses / Total Operating Expenses
            4. Features matching "net income" or "net earnings" → Net Income

            Cash Flow totals:
            1. Features matching "net cash provided by operating activities" or "net cash from operating" → Operating Activities / Net Cash from Operating Activities
            2. Features matching "net cash used in investing activities" or "net cash from investing" → Investing Activities / Net Cash from Investing Activities
            3. Features matching "net cash provided by financing activities" or "net cash from financing" → Financing Activities / Net Cash from Financing Activities

            ROLLUP DETECTION (CRITICAL - PREVENTS DOUBLE-COUNTING):
            {rollup_detection if rollup_detection else "  - Check for combined/rollup features that would cause double-counting"}

            MAPPING GUIDELINES:
            {general_rules}

            SPECIFIC CATEGORY MAPPINGS:
            {specific_mappings}

            EDGE CASES:
            {edge_cases}

            MAPPING EXAMPLES (for reference, not exhaustive):
            {examples_context}

            PRIORITIZATION STRATEGY:
            1. FIRST: Check if feature matches CRITICAL KEYWORD MATCHING RULES above - apply those with confidence
            2. SECOND: Check if the feature is a ROLLUP that combines other separate features - if yes, map to __unmapped__ to prevent double-counting
            3. THIRD: Check if the feature matches any CRITICAL TOTAL/ROLLUP rules above - if yes, map there immediately
            4. FOURTH: Check if feature represents a meaningful financial statement line item
            5. IGNORE IRRELEVANT FEATURES: Skip features that are:
            - Meta-information (filing dates, document IDs, entity information, CIK numbers)
            - Ratios or percentages (unless specifically part of per-share data)
            - Share counts (unless in per-share data section)
            - Duplicative or redundant information
            - Administrative fields
            6. Place ignored features into "__unmapped__" array

            MAPPING PROCESS FOR NON-TOTAL FEATURES:
            1. For each feature, determine if it represents a valid line item for this statement type
            2. If valid, choose the single best leaf category by:
            a. Match to the most specific applicable category
            b. Use mapping examples as guidance (not strict rules)
            c. Prefer specific categories over "Other" categories
            d. If ambiguous, choose the category with the closest accounting meaning
            3. If invalid or unclear, place in "__unmapped__"

            QUALITY OVER COMPLETENESS:
            - It is BETTER to leave a feature unmapped than to force it into an incorrect category
            - An empty array for a line item is acceptable if no relevant feature exists
            - "__unmapped__" should contain meta-information, non-financial items, AND rollup features that would cause double-counting
            - However, features matching CRITICAL KEYWORD RULES should be mapped confidently, not conservatively

            SELF-CHECK (silent before output):
            - Total mapped items across ALL leaf lists (including __unmapped__) equals number of input features
            - No duplicates across leaves
            - All total/rollup features are in their correct "Total ..." leaves, NOT in __unmapped__
            - No rollup/combined features that would cause double-counting in regular categories
            - Features matching CRITICAL KEYWORD RULES are mapped to their specified categories
            - Only relevant line items are in non-__unmapped__ leaves
            - Fix until all checks pass, then output JSON only
            """

        user_content = f"""
            STRUCTURE_JSON:
            {structure_str}

            FEATURES_JSON_ARRAY:
            {features_str}

            CRITICAL ANTI-DOUBLE-COUNTING RULE:
            - Before mapping any feature, scan the full feature list for potential rollups
            - If a feature name combines multiple concepts (e.g., "cash cash equivalents and short term investments"), check if those concepts exist as separate features
            - If the components exist separately, the combined feature is a ROLLUP - map it to __unmapped__ to prevent double-counting
            - Example: "cash cash equivalents and short term investments" should go to __unmapped__ if you see both "cash and cash equivalents" AND "short term investments" as separate features

            APPLY KEYWORD MATCHING RULES:
            - Features containing "before income taxes" or "before taxes" → Income Before Taxes (not __unmapped__)
            - Features exactly matching "nonoperating income expense" → Total Non-Operating Items (not __unmapped__)
            - Features matching "costs and expenses" → __unmapped__ (this is a rollup)

            Return ONLY the updated JSON with features mapped to appropriate categories.
            CRITICAL: Features like "assets", "liabilities", "assets current" MUST go to their Total leaves, NOT __unmapped__.
            Be conservative for non-total line items EXCEPT when they match CRITICAL KEYWORD RULES - those should be mapped confidently.
            """

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        raw = self.chat(messages, cfg)

        # Parse response
        try:
            response_text = raw.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            mapped_structure = json.loads(response_text)

            # Apply statement-specific post-processing fixes as safety net
            mapped_structure = apply_statement_specific_fixes(
                statement_type, mapped_structure, features
            )

            return mapped_structure
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse LLM response as JSON: {e}\nRaw response: {raw}"
            )

    def load_cached_features(
        self,
        cache_path: Path,
    ) -> Dict[str, Any]:
        """Load cached parsed features from JSON file."""
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def save_features_to_cache(
        self,
        features: Dict[str, Any],
        cache_path: Path,
    ) -> None:
        """Save parsed features to JSON cache file."""
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=4)

    def parse_annual_report(
        self,
        pdf_path: str,
        cfg: LLMConfig,
        page_range: tuple[int, int] = (56, 58),
    ) -> dict[str, float]:
        """
        Extract income statement and balance sheet data from PDF annual report.

        Args:
            pdf_path: Path to PDF file
            cfg: LLM configuration
            page_range: Tuple of (start_page, end_page) to extract from (1-indexed)

        Returns:
            Dictionary containing extracted financial metrics and calculated ratios
        """

        # Extract only the specified pages
        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        # pypdf uses 0-indexed pages, convert from 1-indexed input
        for page_num in range(page_range[0] - 1, page_range[1]):
            if page_num < len(reader.pages):
                writer.add_page(reader.pages[page_num])

        # Write to bytes buffer
        pdf_buffer = BytesIO()
        writer.write(pdf_buffer)
        pdf_buffer.seek(0)

        pdf_text = ""
        for page_num in range(page_range[0] - 1, page_range[1]):
            if page_num < len(reader.pages):
                page = reader.pages[page_num]
                pdf_text += page.extract_text() + "\n\n"

        system_content = (
            "You are a financial analyst expert. "
            "Extract specific values from financial statements and calculate the requested ratios. "
            "Return results as a structured JSON object with numerical values only."
        )

        user_content = (
            "Analyze this excerpt from an annual report and provide the following for the latest year available:\n\n"
            "1. Net income (current year)\n"
            "2. Cost-to-income ratio\n"
            "3. Quick ratio\n"
            "4. Debt-to-equity ratio\n"
            "5. Debt-to-assets ratio\n"
            "6. Debt-to-capital ratio\n"
            "7. Debt-to-EBITDA ratio\n"
            "8. Interest coverage ratio\n\n"
            "Return ONLY valid JSON with these exact keys:\n"
            '{"net_income": <value>, "cost_to_income_ratio": <value>, '
            '"quick_ratio": <value>, "debt_to_equity_ratio": <value>, '
            '"debt_to_assets_ratio": <value>, "debt_to_capital_ratio": <value>, '
            '"debt_to_ebitda_ratio": <value>, "interest_coverage_ratio": <value>}\n\n'
            "Net income should be in millions. All ratios should be decimal values "
            "(e.g., 0.65 for 65%).\n\n"
            f"Here is the financial report excerpt:\n\n{pdf_text}"
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        raw = self.chat(messages, cfg)

        try:
            import json

            data = json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            # Try to extract JSON from markdown code blocks
            import re

            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
            else:
                raise ValueError(
                    f"Failed to parse LLM response as JSON: {exc}"
                ) from exc

        required_keys = [
            "net_income",
            "cost_to_income_ratio",
            "quick_ratio",
            "debt_to_equity_ratio",
            "debt_to_assets_ratio",
            "debt_to_capital_ratio",
            "debt_to_ebitda_ratio",
            "interest_coverage_ratio",
        ]

        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(f"LLM response missing keys: {missing}")

        # Pretty print the extracted data
        self._pretty_print_financial_data(data)

        return data

    def _pretty_print_financial_data(self, data: dict) -> None:
        """Pretty print extracted financial data."""

        def format_value(val, fmt_type="ratio", show_pct=False):
            """Format a value or return N/A if None."""
            if val is None:
                return "N/A".rjust(12 if not show_pct else 28)
            if fmt_type == "currency":
                # Convert millions to dollars and use format_money
                return format_money(val * 1_000_000).rjust(12)
            elif show_pct:
                return f"{val:>12.4f} ({val * 100:>6.2f}%)"
            else:
                return f"{val:>12.4f}"

        lines = ["=" * 70, "Extracted Financial Data", "=" * 70]

        lines.append("\nINCOME METRICS:")
        lines.append("-" * 70)
        lines.append(
            f"  Net Income:                    "
            f"{format_value(data.get('net_income'), 'currency')}"
        )

        lines.append("\nFINANCIAL RATIOS:")
        lines.append("-" * 70)

        financial_ratios = [
            ("Cost-to-Income Ratio:", "cost_to_income_ratio", True),
            ("Quick Ratio:", "quick_ratio", False),
            ("Interest Coverage Ratio:", "interest_coverage_ratio", False),
        ]

        for label, key, show_pct in financial_ratios:
            lines.append(
                f"  {label:30} {format_value(data.get(key), show_pct=show_pct)}"
            )

        lines.append("\nLEVERAGE RATIOS:")
        lines.append("-" * 70)

        leverage_ratios = [
            ("Debt-to-Equity Ratio:", "debt_to_equity_ratio", True),
            ("Debt-to-Assets Ratio:", "debt_to_assets_ratio", True),
            ("Debt-to-Capital Ratio:", "debt_to_capital_ratio", True),
            ("Debt-to-EBITDA Ratio:", "debt_to_ebitda_ratio", False),
        ]

        for label, key, show_pct in leverage_ratios:
            lines.append(
                f"  {label:30} {format_value(data.get(key), show_pct=show_pct)}"
            )

        lines.append("=" * 70)

        print("\n".join(lines))


if __name__ == "__main__":
    client = LLMClient()
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-5-mini",
    )

    data = client.parse_annual_report(
        pdf_path="/scratch/datasets/jpm/gm_annual_report.pdf",
        cfg=llm_config,
        page_range=(61, 64),
    )
