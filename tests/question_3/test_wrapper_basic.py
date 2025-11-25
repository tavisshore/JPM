import pandas as pd
import numpy as np

from choice_learn_ext.models.deep_context.deep_halo_core import DeepHaloChoiceModel


def test_wrapper_fit_predict():
    df = pd.DataFrame({
        "available": [
            [1,1,1],
            [1,1,1],
            [1,1,1],
        ],
        "item_ids": [
            [0,1,2],
            [0,1,2],
            [0,1,2],
        ],
        "choice": [0,1,2],
    })

    model = DeepHaloChoiceModel(num_items=3, epochs=3, batch_size=2, lr=1e-2)
    model.fit(df)

    probs = model.predict_proba(df)
    assert probs.shape == (3,3)

    preds = model.predict(df)
    assert preds.shape == (3,)
