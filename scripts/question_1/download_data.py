import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from tqdm import tqdm

from jpm.question_1 import Config, EdgarData

# S&P 500 companies
sp500 = """
AAPL MSFT GOOGL AMZN NVDA META TSLA BRK.B UNH XOM JNJ JPM V PG MA HD CVX ABBV MRK AVGO
COST PEP ADBE LLY WMT TMO CSCO MCD ACN CRM ABT NFLX DHR TXN NKE DIS ORCL VZ INTC
CMCSA PFE PM NEE WFC UPS COP RTX IBM BA QCOM AMD INTU NOW CAT GS MS SPGI LOW AXP
ISRG HON BKNG TJX BLK AMAT GILD SYK DE LRCX VRTX PLD MMC MDLV CI C ADI REGN SCHW
ZTS ETN CB SO MU AON SLB BSX FISV BMY DUK ITW BDX PNC CME APD EQIX USB ICE MCO
MMM GE COF NSC TGT HCA PYPL PH EOG MAR MO EMR NOC CSX TFC WM PSA WELL KLAC KMI
APH CCI ROP SHW HUM ORLY GM ADSK NXPI F PCAR AIG MET ALL AEP AJG ROST KMB SRE
EW CTAS CARR MSCI PAYX IDXX AFL DD FCX FTNT A O PRU CTVA ODFL RSG YUM KDP FAST
CDNS SNPS GWW MNST CHTR VRSK EL MCHP RCL CPRT GD CMG ANSS DAL AME IT BIIB LHX
""".split()

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
    futures = {executor.submit(process_ticker, ticker): ticker for ticker in sp500}

    results = {}
    errors = {}

    with tqdm(total=len(sp500), desc="Downloading", unit="ticker") as pbar:
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
print(f"Successfully processed: {len(results)}/{len(sp500)}")
if errors:
    print(f"Failed tickers: {', '.join(errors.keys())}")
