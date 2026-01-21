import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from tqdm import tqdm

from jpm.config import Config
from jpm.question_1 import EdgarData, get_args
from jpm.utils import get_tickers

args = get_args()

data_tickers = get_tickers(args.industry, length=args.total_tickers)

rate_limiter = Semaphore(10)


def process_ticker(ticker):
    """Process a single ticker with rate limiting."""
    with rate_limiter:
        try:
            config = copy.deepcopy(Config())
            config.data.ticker = ticker
            data = EdgarData(config=config, overwrite=False, verbose=False)
            time.sleep(0.1)
            return ticker, data, None
        except Exception as e:
            return ticker, None, str(e)


max_workers = 4

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(process_ticker, ticker): ticker for ticker in data_tickers
    }

    results = {}
    errors = {}

    with tqdm(total=len(data_tickers), desc="Downloading", unit="ticker") as pbar:
        for future in as_completed(futures):
            ticker = futures[future]
            ticker_result, data, error = future.result()

            if error:
                tqdm.write(f"❌ {ticker_result}: {error}")
                errors[ticker_result] = error
            else:
                results[ticker_result] = data

            pbar.update(1)
            pbar.set_postfix_str(f"✅ {len(results)} | ❌ {len(errors)}")

print(f"\n{'=' * 60}")
print(f"Successfully processed: {len(results)}/{len(data_tickers)}")
if errors:
    print(f"Failed tickers: {', '.join(errors.keys())}")
