import pandas as pd


def get_predict_prompt(
    adjust: bool, data_csv: str, prediction: pd.DataFrame | None = None
) -> list[dict[str, str]]:
    if adjust:
        assert prediction is not None, "Prediction DataFrame required for adjust=True"
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
    return messages
