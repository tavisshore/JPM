from __future__ import annotations

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
        prediction=None,
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
