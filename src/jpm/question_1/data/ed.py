import os
from pathlib import Path

import edgar
import numpy as np
import pandas as pd
import tensorflow as tf
from edgar import Company
from edgar.xbrl import XBRLS
from sklearn.preprocessing import StandardScaler

from src.jpm.question_1.data.utils import (
    bs_identity,
    build_windows,
    get_bs_structure,
    get_cf_structure,
    get_is_structure,
    get_leaf_values,
    get_targets,
    xbrl_to_snake,
)

# SEC requires user identification via email
email = os.getenv("EDGAR_EMAIL")
if not email:
    raise ValueError(
        "EDGAR_EMAIL environment variable not set"
        "set with 'export EDGAR_EMAIL='example@jpm.com'"
    )

edgar.set_identity(email)

# Dates where companies seem to change reporting structure significantly
format_limit = {
    "AAPL": "2018-12",
}


class EdgarDataLoader:
    def __init__(
        self,
        ticker="AAPL",
        periods=40,
        cache_dir="/Users/tavisshore/Desktop/HK/data",
        target="full",
    ) -> None:
        self.cache_statement = Path(f"{cache_dir}/{ticker}.parquet")
        self.ticker = ticker
        self.periods = periods
        self.lookback, self.horizon = 4, 1
        self.batch_size = 32
        self.target_type = target
        self.bs_structure = get_bs_structure(ticker=ticker)

        self.company = Company(ticker)
        self.feat_stat = {}
        self.create_dataset()

    def create_dataset(self) -> None:
        self.bs_keys = get_bs_structure(ticker=self.ticker, flatten=True)

        if self.cache_statement.exists():
            self.data = pd.read_parquet(self.cache_statement)
        else:
            self.create_statements()

        # Actually names of features in self.data to column indices
        self.feat_to_idx = {n: i for i, n in enumerate(self.data.columns.tolist())}

        # Maybe - check accounting identities now, before training
        bs_identity(self.data, ticker=self.ticker)
        # TODO - Add more checks

        # Now convert into tensorflow training data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.data.values.astype("float32"))
        self.feat_stat["mean"] = scaler.mean_.astype("float32")
        self.feat_stat["std"] = scaler.scale_.astype("float32")

        if self.target_type != "full":
            tar = get_targets(mode=self.target_type, ticker=self.ticker)
            self.tgt_indices = [
                self.feat_to_idx[t] for t in tar if t in self.feat_to_idx
            ]
        else:
            self.tgt_indices = list(range(len(self.data.columns)))
        self.map_features()
        # Create separate windows for individual companies + concatenate
        # X_windows_list, y_windows_list = [], []
        X_train, y_train, X_test, y_test = build_windows(
            X=X_scaled,
            lookback=self.lookback,
            horizon=self.horizon,
            tgt_indices=self.tgt_indices,
        )
        # X_windows_list.append(Xw)
        # y_windows_list.append(yw)

        # X_all = np.concatenate(X_windows_list, axis=0)  # (N, lookback, F)
        # y_all = np.concatenate(y_windows_list, axis=0)  # (N, F)
        self.num_features = X_train.shape[-1]  # Input dim
        self.num_targets = len(self.tgt_indices)  # Output dim
        self.train_dataset = (
            tf.data.Dataset.from_tensor_slices(
                (X_train.astype("float32"), y_train.astype("float32"))
            )
            .shuffle(len(X_train))
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        self.test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(
            self.batch_size
        )

    def map_features(self):
        """
        With the bs, is, and cf structures known, map feature names to indices
        For loss, calculations etc.
        """
        # Start with A = L + E structure
        self.feature_mappings = {}
        self.feature_mappings["assets"] = [
            self.feat_to_idx[k] for k in get_leaf_values(self.bs_structure, "assets")
        ]
        self.feature_mappings["liabilities"] = [
            self.feat_to_idx[k]
            for k in get_leaf_values(self.bs_structure, "liabilities")
        ]
        self.feature_mappings["equity"] = [
            self.feat_to_idx[k] for k in get_leaf_values(self.bs_structure, "equity")
        ]

        # Maybe make sub-categories lower weighted losses later?
        self.feature_mappings["current_assets"] = [
            self.feat_to_idx[k]
            for k in get_leaf_values(self.bs_structure, "current_assets")
        ]
        self.feature_mappings["non_current_assets"] = [
            self.feat_to_idx[k]
            for k in get_leaf_values(self.bs_structure, "non_current_assets")
        ]
        self.feature_mappings["current_liabilities"] = [
            self.feat_to_idx[k]
            for k in get_leaf_values(self.bs_structure, "current_liabilities")
        ]
        self.feature_mappings["non_current_liabilities"] = [
            self.feat_to_idx[k]
            for k in get_leaf_values(self.bs_structure, "non_current_liabilities")
        ]
        # Adding mappings here

    def create_statements(self) -> None:
        self.filings = self.company.get_filings(form="10-Q")
        self.xbrls = XBRLS.from_filings(self.filings)

        # Process each statement
        self.bs_df = self._process_statement(
            stmt=self.xbrls.statements.balance_sheet(max_periods=self.periods),
            kind="balance sheet",
            needed_cols=self.bs_keys,
        )
        self.is_df = self._process_statement(
            stmt=self.xbrls.statements.income_statement(max_periods=self.periods),
            kind="income statement",
            needed_cols=get_is_structure(ticker=self.ticker, flatten=True),
        )
        self.cf_df = self._process_statement(
            stmt=self.xbrls.statements.cashflow_statement(max_periods=self.periods),
            kind="cash flow statement",
            needed_cols=get_cf_structure(ticker=self.ticker, flatten=True),
        )

        # Align all statements on common dates (monthly periods)
        for attr in ("bs_df", "is_df", "cf_df"):
            df = getattr(self, attr)
            df.index = df.index.to_period("M")

        self.data = pd.concat(
            [self.bs_df, self.is_df, self.cf_df], axis=1, join="inner"
        )
        self.data.to_parquet(self.cache_statement)

    def _process_statement(
        self,
        stmt,
        kind: str,
        needed_cols: list[str],
    ) -> pd.DataFrame:
        """
        Common XBRL → tidy DataFrame pipeline.
        """
        if stmt is None:
            raise ValueError(f"No {kind} found for {self.ticker}")

        df = stmt.to_dataframe()

        # Split meta vs date cols
        meta_cols = [c for c in ("label", "concept") if c in df.columns]
        date_cols = [c for c in df.columns if c not in meta_cols]

        # concept x dates -> dates x concept
        wide = df.set_index("concept")[date_cols].T

        # Index normalisation
        wide.index = pd.to_datetime(wide.index)
        wide = wide.sort_index()
        wide.index.name = "period_end"

        # Drop records before format change
        if self.ticker in format_limit:
            wide = wide[wide.index >= format_limit[self.ticker]]

        # Normalise column names, collapse duplicates, clean NaNs
        wide.columns = [xbrl_to_snake(col) for col in wide.columns]
        collapsed = wide.T.groupby(level=0).first().T  # collapse duplicate concepts
        collapsed = collapsed.replace(r"^\s*$", np.nan, regex=True)
        collapsed = collapsed.dropna(axis=1, how="all")
        collapsed = collapsed.fillna(0)

        # Only keep necessary columns (will raise if missing -> good fail-fast)
        # print(collapsed.columns)
        return collapsed[needed_cols]


if __name__ == "__main__":
    loader = EdgarDataLoader(
        ticker="AAPL", cache_dir="/Users/tavisshore/Desktop/HK/data"
    )
