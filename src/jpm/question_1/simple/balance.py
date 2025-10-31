from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, cast

import tensorflow as tf

from src.jpm.question_1.misc import to_tensor


@dataclass
class BalanceSheet:
    # Assets
    cash_and_equivalents: tf.Tensor
    short_term_investments: tf.Tensor
    accounts_receivable: tf.Tensor
    inventory: tf.Tensor
    other_current_assets: tf.Tensor
    total_current_assets: tf.Tensor
    pp_and_e: tf.Tensor
    goodwill_intangibles: tf.Tensor
    other_non_current_assets: tf.Tensor
    total_assets: tf.Tensor

    # Liabilities
    accounts_payable: tf.Tensor
    short_term_debt: tf.Tensor
    other_current_liabilities: tf.Tensor
    total_current_liabilities: tf.Tensor
    long_term_debt: tf.Tensor
    other_non_current_liabilities: tf.Tensor
    total_liabilities: tf.Tensor

    # Equity
    common_stock: tf.Tensor
    retained_earnings: tf.Tensor
    aoci: tf.Tensor
    treasury_stock: tf.Tensor
    total_shareholder_equity: tf.Tensor

    validity_threshold: tf.Tensor = field(
        default_factory=lambda: cast(tf.Tensor, tf.constant(1e-4, dtype=tf.float32))
    )

    def __post_init__(self):
        for f in self.__dataclass_fields__:
            setattr(self, f, to_tensor(getattr(self, f)))

    def validate(self) -> Dict[str, tf.Tensor]:
        errs: Dict[str, tf.Tensor] = {}
        cash_like = tf.add(self.cash_and_equivalents, self.short_term_investments)
        calc_tca = tf.add(
            tf.add(cash_like, self.accounts_receivable),
            tf.add(self.inventory, self.other_current_assets),
        )
        errs["total_current_assets"] = tf.math.abs(
            tf.subtract(self.total_current_assets, calc_tca)
        )

        calc_ta = tf.add(
            tf.add(self.total_current_assets, self.pp_and_e),
            tf.add(self.goodwill_intangibles, self.other_non_current_assets),
        )
        errs["total_assets"] = tf.math.abs(tf.subtract(self.total_assets, calc_ta))

        calc_tcl = tf.add(
            tf.add(self.accounts_payable, self.short_term_debt),
            self.other_current_liabilities,
        )
        errs["total_current_liabilities"] = tf.math.abs(
            tf.subtract(self.total_current_liabilities, calc_tcl)
        )

        calc_tl = tf.add(
            tf.add(self.total_current_liabilities, self.long_term_debt),
            self.other_non_current_liabilities,
        )
        errs["total_liabilities"] = tf.math.abs(
            tf.subtract(self.total_liabilities, calc_tl)
        )

        calc_equity = tf.add(
            tf.add(self.common_stock, self.retained_earnings),
            tf.add(self.aoci, self.treasury_stock),
        )
        errs["total_shareholder_equity"] = tf.math.abs(
            tf.subtract(self.total_shareholder_equity, calc_equity)
        )

        errs["balance_equation"] = tf.math.abs(
            tf.subtract(
                self.total_assets,
                tf.add(self.total_liabilities, self.total_shareholder_equity),
            )
        )
        return errs

    @property
    def is_valid(self) -> tf.Tensor:
        vals = [
            tf.convert_to_tensor(v, dtype=tf.float32) for v in self.validate().values()
        ]
        return tf.reduce_all(tf.math.less(tf.stack(vals), self.validity_threshold))

    def to_tensor_dict(self) -> Dict[str, tf.Tensor]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


if __name__ == "__main__":
    from src.jpm.question_1.data.yf import FinanceIngestor

    ingestor = FinanceIngestor()
