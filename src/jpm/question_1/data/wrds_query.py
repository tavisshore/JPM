import os
from datetime import datetime

import pandas as pd
import wrds


def get_quarterly_credit_ratings(
    ticker,
    start_date="2014-01-01",
    end_date=None,
):
    """
    Get quarterly historical Moody's credit ratings for a company from WRDS.

    Parameters:
    -----------
    ticker : str
        Company ticker symbol (e.g., 'AAPL')
    start_date : str
        Start date in 'YYYY-MM-DD' format (default: 10 years ago)
    end_date : str
        End date in 'YYYY-MM-DD' format (default: today)

    Returns:
    --------
    pd.DataFrame
        Quarterly credit ratings with columns: quarter, \
            rating_date, long_term_rating, outlook

    Environment Variables:
    ----------------------
    WRDS_USERNAME : str
        WRDS username (required)
    """

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    wrds_username = os.getenv("WRDS_USERNAME")
    if not wrds_username:
        raise ValueError("WRDS_USERNAME environment variable not set")

    db = wrds.Connection(wrds_username=wrds_username)

    query = f"""
    SELECT c.companyname, c.tickersymbol, r.ratingdate,
           r.ltrating as long_term_rating, r.ltoutlook as outlook,
           r.strating as short_term_rating
    FROM ciq.wrds_rating r
    JOIN ciq.wrds_company c ON r.companyid = c.companyid
    WHERE c.tickersymbol = '{ticker}'
      AND r.ratingagency = 'Moody''s'
      AND r.ratingdate >= '{start_date}'
      AND r.ratingdate <= '{end_date}'
    ORDER BY r.ratingdate
    """

    ratings = db.raw_sql(query)
    db.close()

    if ratings.empty:
        print(f"No ratings found for {ticker}")
        return pd.DataFrame()

    ratings["ratingdate"] = pd.to_datetime(ratings["ratingdate"])
    ratings["quarter"] = ratings["ratingdate"].dt.to_period("Q")

    quarterly = (
        ratings.sort_values("ratingdate").groupby("quarter").last().reset_index()
    )

    quarterly = quarterly[
        ["quarter", "ratingdate", "long_term_rating", "outlook", "companyname"]
    ]
    quarterly.columns = [
        "quarter",
        "rating_date",
        "long_term_rating",
        "outlook",
        "company_name",
    ]

    return quarterly


if __name__ == "__main__":
    df = get_quarterly_credit_ratings("AAPL", start_date="2014-01-01")
    print(df)
