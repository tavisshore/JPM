from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from io import StringIO
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI
from pandas import DataFrame

from jpm.question_1.config import LLMConfig


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
            assert (
                prediction is not None
            ), "Prediction DataFrame required for adjust=True"
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
        structure_json: Dict[str, Any],
        cfg: LLMConfig,
    ) -> Dict[str, Any]:
        structure_str = json.dumps(
            structure_json,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        features_str = json.dumps(features, ensure_ascii=False)

        system_content = """
                You are a deterministic mapper from feature strings to a fixed balance-sheet JSON taxonomy.

                Definitions:
                - A "leaf" is any key whose value is a JSON array (list). Only leaves may be modified.
                - Non-leaves are JSON objects (dicts) and must not be modified.

                Hard requirements:
                - Output must be a single valid JSON object.
                - The output object MUST keep identical keys and nesting as the input structure (no new keys; no renamed keys).
                - Only append items to existing leaf arrays.
                - Feature strings must be copied verbatim (exact characters; no normalization).

                Mapping strategy:
                1. PRIORITIZE USEFUL FEATURES: Only map features that represent meaningful balance sheet line items.
                2. IGNORE IRRELEVANT FEATURES: Skip features that are:
                - Meta-information (filing dates, document IDs, entity information, CIK numbers)
                - Ratios or percentages
                - Duplicative or redundant information
                - Administrative fields
                - Non-balance-sheet items
                3. Place ignored features into "__unmapped__" array.

                Totals / rollups routing (apply FIRST for relevant features):
                - If a feature is a rollup/total/subtotal, map it to the most specific matching "Total ..." leaf:
                * "assets current" or "total current assets" -> Assets / Current / Total Current Assets
                * "assets noncurrent" or "total non-current assets" -> Assets / Non-Current / Total Non-Current Assets
                * "assets" or "total assets" -> Assets / Total Assets
                * "liabilities current" or "total current liabilities" -> Liabilities / Current / Total Current Liabilities
                * "liabilities noncurrent" or "total non-current liabilities" -> Liabilities / Non-Current / Total Non-Current Liabilities
                * "liabilities" or "total liabilities" -> Liabilities / Total Liabilities
                * "stockholders equity" or "shareholders equity" or "total equity" -> Equity / Total Equity
                * "liabilities and stockholders equity" (or similar) -> Totals / Total Liabilities and Equity

                Share-count routing (for relevant features):
                - Features containing "shares issued" -> Equity / Common Stock / Shares Issued (or Preferred if explicitly preferred)
                - Features containing "shares outstanding" -> Equity / Common Stock / Shares Outstanding (or Preferred if explicitly preferred)
                - Features containing "common stock" without shares -> Equity / Common Stock / Amount
                - Features containing "preferred stock" without shares -> Equity / Preferred Stock / Amount

                Normal mapping rule (for relevant features only):
                - Choose the single best leaf by accounting meaning.
                - If ambiguous, choose the best accounting meaning; if still tied, choose the leaf whose full path is alphabetically first.
                - Preserve the input order of features within each leaf list (stable assignment).
                - If a feature does NOT clearly represent a balance sheet line item, place it in "__unmapped__".

                Quality over completeness:
                - It is BETTER to leave a feature unmapped than to force it into an incorrect category.
                - An empty array for a balance sheet line item is acceptable if no relevant feature exists.
                - "__unmapped__" may contain many items - this is expected and correct.

                Self-check (silent before output):
                - Total mapped items across ALL leaf lists (including __unmapped__) equals number of input features.
                - No duplicates across leaves.
                - Only balance-sheet-relevant features are in non-__unmapped__ leaves.
                Fix until all checks pass, then output JSON only.
                """

        user_content = f"""
        STRUCTURE_JSON:
        {structure_str}

        FEATURES_JSON_ARRAY:
        {features_str}

        Return ONLY the updated JSON. Be conservative - when in doubt, use __unmapped__.
        """

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        raw = self.chat(messages, cfg)

        try:
            parsed = json.loads(raw)
        except Exception as exc:
            raise ValueError(f"Failed to parse LLM response as JSON: {exc}") from exc

        # Print the features not sorted:
        # sorted_values = leaf_list_values(parsed)
        # not_sorted = [f for f in features if f not in sorted_values]
        # print(f"\nNot Sorted: {not_sorted}\n")

        return parsed


if __name__ == "__main__":
    client = LLMClient()

    df_example = pd.DataFrame(
        {
            "revenue": [100, 120, 150, 170],
            "ebitda": [20, 25, 30, 32],
        },
        index=pd.period_range(start="2022Q1", periods=4, freq="Q"),
    )

    cfg_oa = LLMConfig(
        provider="openai",
        model="gpt-5-nano",
        temperature=0.2,
        max_tokens=512,
    )

    print("=== Forecast next quarter (demo) ===")
    try:
        print(client.forecast_next_quarter(df_example, cfg_oa))
    except Exception as exc:
        print(f"Forecast call failed: {exc}")
        print(f"Forecast call failed: {exc}")
