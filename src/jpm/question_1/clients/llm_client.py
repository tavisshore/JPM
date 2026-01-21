from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI
from pandas import DataFrame
from pypdf import PdfReader
from tqdm import tqdm

from jpm.config.question_1 import Config, LLMConfig
from jpm.question_1.clients.prompts import (
    get_company_name_prompt,
    get_predict_prompt,
    get_report_prompt,
    get_statement_prompt,
    get_ticker_prompt,
)
from jpm.question_1.clients.utils import (
    apply_statement_specific_fixes,
    get_fx_rate,
    parse_llm_json_response,
)
from jpm.question_1.misc import format_money


class LLMClient:
    """
    Client for LLM-based financial data analysis and extraction.

    Provides integration with language models for parsing financial statements,
    forecasting metrics, and converting between company names and ticker symbols.
    Supports OpenAI models for various financial analysis tasks.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the LLM client with API credentials.

        Parameters
        ----------
        openai_api_key : Optional[str], optional
            OpenAI API key for authentication. If not provided, will attempt to
            read from OPENAI_API_KEY environment variable.

        Raises
        ------
        RuntimeError
            If no API key is provided and OPENAI_API_KEY is not set.
        """
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            raise RuntimeError(
                "OpenAI API key is missing. "
                "Set OPENAI_API_KEY or pass openai_api_key explicitly."
                "Otherwise - using offline data only."
            )

        self._openai: OpenAI = OpenAI(api_key=openai_api_key)

    def chat(
        self, messages: List[Dict[str, Any]], cfg: LLMConfig, verbose: bool = False
    ) -> str:
        """
        Send a chat request to the configured LLM provider.

        Parameters
        ----------
        messages : List[Dict[str, Any]]
            List of message dictionaries containing role and content keys.
        cfg : LLMConfig
            Configuration specifying the provider and model to use.
        verbose : bool, default=False
            If True, display a spinner while waiting for the response.

        Returns
        -------
        str
            The text content of the LLM's response.

        Raises
        ------
        ValueError
            If the provider specified in cfg is not supported.
        """
        if cfg.provider == "openai":
            return self._chat_openai(messages, cfg, verbose)
        else:
            raise ValueError(f"Unknown provider: {cfg.provider}")

    def _chat_openai(
        self, messages: List[Dict[str, Any]], cfg: LLMConfig, verbose: bool = False
    ) -> str:
        """
        Internal method to send chat request to OpenAI API.

        Parameters
        ----------
        messages : List[Dict[str, Any]]
            List of message dictionaries containing role and content keys.
        cfg : LLMConfig
            Configuration specifying the model and other parameters.
        verbose : bool, default=False
            If True, display a spinner while waiting for the response.

        Returns
        -------
        str
            The text content of the OpenAI response.

        Raises
        ------
        RuntimeError
            If the OpenAI client is not initialized.
        """
        if self._openai is None:
            raise RuntimeError("OpenAI client not initialised (missing API key).")
        if verbose:
            with self._spinner(f"Waiting for {cfg.provider} {cfg.model} response"):
                if "nano" not in cfg.model:
                    resp = self._openai.chat.completions.create(
                        model=cfg.model,
                        messages=messages,
                    )
                else:
                    resp = self._openai.chat.completions.create(
                        model=cfg.model,
                        messages=messages,
                    )
        else:
            if "nano" not in cfg.model:
                resp = self._openai.chat.completions.create(
                    model=cfg.model,
                    messages=messages,
                )
            else:
                resp = self._openai.chat.completions.create(
                    model=cfg.model,
                    messages=messages,
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
        verbose: bool = False,
    ) -> DataFrame:
        """
        Forecast next quarter's financial metrics using LLM.

        Uses historical quarterly financial data to predict the next quarter's
        values for specified financial metrics. Optionally adjusts predictions
        based on a provided prediction DataFrame.

        Parameters
        ----------
        history : DataFrame
            Historical quarterly financial data with datetime index.
        cfg : LLMConfig
            Configuration for the LLM provider and model.
        prediction : DataFrame or None, default=None
            Optional prediction DataFrame to adjust against.
        feature_columns : Optional[List[str]], default=None
            List of column names to forecast. If None, uses all columns.
        max_rows : int, default=4
            Maximum number of recent rows to include in the prompt.
        adjust : bool, default=False
            If True, instructs LLM to adjust based on prediction parameter.
        verbose : bool, default=False
            If True, display a spinner during API call.

        Returns
        -------
        DataFrame
            Single-row DataFrame containing forecasted values for next quarter.

        Raises
        ------
        ValueError
            If history is None, empty, or missing required columns.
            If LLM response cannot be parsed as CSV or is missing columns.
        """
        if history is None or history.empty:
            raise ValueError("DataFrame must contain quarterly financial data.")

        target_cols = feature_columns or list(history.columns)
        missing = [col for col in target_cols if col not in history.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

        recent = history.sort_index().tail(max_rows)[target_cols]
        data_csv = recent.to_csv(index=True)

        prompt = get_predict_prompt(adjust, data_csv, prediction)

        raw = self.chat(prompt, cfg, verbose)

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
        prompt = get_statement_prompt(statement_type, features)

        raw = self.chat(prompt, cfg)

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

    def save_features_to_cache(
        self, features: Dict[str, Any], cache_path: Path
    ) -> None:
        """Save parsed features to JSON cache file."""
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=4)

    def parse_annual_report(self, config: Config) -> dict[str, float] | None:
        """
        Parse annual report PDF to extract financial metrics and ratios.

        Extracts financial data from specified pages of a PDF annual report,
        converts values to USD if needed, and calculates key financial ratios
        including leverage, liquidity, and profitability metrics.

        Parameters
        ----------
        config : Config
            Configuration object containing report path and page ranges.

        Returns
        -------
        dict[str, float] or None
            Dictionary containing extracted financial metrics and calculated ratios:
            - net_income : Net income in USD
            - Cost_to_Income : Operating expenses / Revenue
            - Quick_Ratio : (Current Assets - Inventory) / Current Liabilities
            - Debt_to_Equity : Total Debt / Equity
            - Debt_to_Assets : Total Debt / Total Assets
            - Total_Debt_to_Total_Capital : Total Debt / (Total Debt + Equity)
            - Debt_to_EBITDA : Total Debt / EBITDA
            - Interest_Coverage : EBIT / Interest Expense
            - original_currency : Currency from the report
            - exchange_rate : FX rate used for USD conversion
            - report_date : Fiscal year end date
            Returns None if parsing or extraction fails.
        """
        pdf_path = config.data.get_report_path()
        page_range = config.data.get_report_pages()

        reader = PdfReader(pdf_path)
        pdf_text = ""
        for start, end in page_range:
            for page_num in range(start - 1, end):
                if page_num < len(reader.pages):
                    page = reader.pages[page_num]
                    pdf_text += page.extract_text() + "\n\n"

        prompt = get_report_prompt(pdf_text)

        try:
            stage1 = self._openai.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                seed=42,
                max_tokens=2048,
            )

            content = stage1.choices[0].message.content.strip()
            content = re.sub(r":\s*(\d+),(\d+)", r": \1\2", content)

            raw_values = json.loads(content)

            CA = float(raw_values["current_assets"])
            INV = (
                float(raw_values["inventories"])
                if raw_values["inventories"] is not None
                else 0
            )  # TODO: Goog report doesn't disclose, raise or solve
            CL = float(raw_values["current_liabilities"])
            TD = float(raw_values["financial_liabilities_noncurrent"]) + float(
                raw_values["financial_liabilities_current"]
            )
            EQ = float(raw_values["equity"])
            TA = float(raw_values["total_assets"])
            NI = float(raw_values["net_income"])
            EBIT = float(raw_values["ebit"])
            IE = float(raw_values["interest_expense"])
            REV = float(raw_values["revenue"])
            OE = float(raw_values["operating_expenses"])
            DA = float(raw_values["depreciation_amortization"])

            currency = raw_values["currency"]
            fiscal_year_end = raw_values["fiscal_year_end"]

            fx_rate = get_fx_rate(currency, fiscal_year_end)
            EBITDA = EBIT + IE + DA

            result = {
                # In bonus question
                "net_income": round(NI * fx_rate, 2),
                # Ratios (names match training data feature columns)
                "cost_to_income": round(OE / REV, 4) if REV != 0 else None,
                "quick_ratio": round((CA - INV) / CL, 4) if CL != 0 else None,
                "debt_to_equity": round(TD / EQ, 4) if EQ != 0 else None,
                "debt_to_assets": round(TD / TA, 4) if TA != 0 else None,
                "debt_to_capital": round(TD / (TD + EQ), 4) if (TD + EQ) != 0 else None,
                "debt_to_ebitda": round(TD / EBITDA, 4) if EBITDA != 0 else None,
                "interest_coverage": round(EBIT / IE, 4) if IE != 0 else None,
                "original_currency": currency,
                "exchange_rate": fx_rate,
                "report_date": fiscal_year_end,
            }
            self._pretty_print_financial_data(result)

        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Content: {content[:1000]}")
            result = None

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            result = None

        return result

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
            ("Cost-to-Income Ratio:", "cost_to_income", True),
            ("Quick Ratio:", "quick_ratio", False),
            ("Interest Coverage Ratio:", "interest_coverage", False),
        ]

        for label, key, show_pct in financial_ratios:
            lines.append(
                f"  {label:30} {format_value(data.get(key), show_pct=show_pct)}"
            )

        lines.append("\nLEVERAGE RATIOS:")
        lines.append("-" * 70)

        leverage_ratios = [
            ("Debt-to-Equity Ratio:", "debt_to_equity", True),
            ("Debt-to-Assets Ratio:", "debt_to_assets", True),
            ("Debt-to-Capital Ratio:", "debt_to_capital", True),
            ("Debt-to-EBITDA Ratio:", "debt_to_ebitda", False),
        ]

        for label, key, show_pct in leverage_ratios:
            lines.append(
                f"  {label:30} {format_value(data.get(key), show_pct=show_pct)}"
            )

        lines.append("=" * 70)

        print("\n".join(lines))

    def company_name_to_ticker(
        self,
        company_names: list,
        cfg: LLMConfig,
    ) -> dict:
        """
        Convert company names to stock ticker symbols using LLM.

        Processes company names in batches to retrieve corresponding stock ticker
        symbols. Falls back to individual processing if batch processing fails.

        Parameters
        ----------
        company_names : list
            List of company name strings to convert to tickers.
        cfg : LLMConfig
            Configuration for the LLM provider and model.

        Returns
        -------
        dict
            Dictionary mapping company names to ticker symbols.
            Values may be None if ticker cannot be determined.
        """
        batch_size = 25
        all_results = {}
        num_batches = (len(company_names) + batch_size - 1) // batch_size

        for i in tqdm(
            range(0, len(company_names), batch_size),
            desc="Processing company batches",
            total=num_batches,
        ):
            batch = company_names[i : i + batch_size]
            try:
                prompt = get_ticker_prompt(batch)
                response = self.chat(prompt, cfg)
                response = parse_llm_json_response(response)
                all_results.update(response)
                print(response)
            except Exception as e:
                print(f"\nBatch {i // batch_size} failed: {e}")
                for name in tqdm(batch, desc="Fallback individual calls", leave=False):
                    try:
                        prompt = get_ticker_prompt([name])
                        response = self.chat(prompt, cfg)
                        response = parse_llm_json_response(response)
                        all_results[name] = response
                    except:
                        all_results[name] = None

        return response

    def ticker_to_company_name(
        self,
        ticker: str,
        cfg: LLMConfig,
    ) -> dict:
        """
        Convert stock ticker symbol to company name variations using LLM.

        Retrieves various name forms and variations for a given ticker symbol,
        which is useful for matching against different data sources.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol to convert.
        cfg : LLMConfig
            Configuration for the LLM provider and model.

        Returns
        -------
        dict
            Dictionary containing ticker and associated company name variations.
        """
        prompt = get_company_name_prompt([ticker])
        response = self.chat(prompt, cfg)
        result = parse_llm_json_response(response)
        name_variations = result.get(ticker, [])
        # response = parse_llm_list_response(response[ticker])
        return name_variations
