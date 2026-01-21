import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from tqdm import tqdm

from jpm.config import Config
from jpm.question_1 import EdgarData, get_args

data_tickers = """
AAPL MSFT GOOGL AMZN NVDA META TSLA JPM BAC WFC C GS MS USB PNC BK TFC
XOM CVX COP SLB EOG MPC PSX VLO HAL OXY DVN FANG HES MRO APA BKR
JNJ PFE MRK ABBV LLY BMY GILD AMGN REGN VRTX BIIB ZTS ISRG BSX MDT ABT TMO DHR
WMT TGT HD LOW COST CVS WBA DG DLTR ROST TJX BBY EBAY
KO PEP MDLZ GIS K MO PM STZ TAP BF.B CAG CPB SJM HSY
T VZ TMUS CMCSA CHTR DIS NFLX PARA FOX FOXA WBD
BA RTX LMT GD NOC HON TDG HWM LDOS
CAT DE MMM ETN EMR ITW ROK DOV PH IR XYL
UNH CI HUM ANTM CNC CVS ELV MOH
MA V AXP DFS COF SYF
F GM TSLA STLA TM HMC
UPS FDX
AMT CCI EQIX PLD SPG O VICI AVB EQR MAA ESS UDR CPT
SO DUK NEE D AEP EXC ED SRE XEL PCG WEC ES PEG ETR FE CNP CMS NI DTE
KR SYY ADM BG TSN HRL INGR
NEM FCX SCCO GOLD AA ALB MP FMC CF MOS APD LIN ECL NUE STLD CLF X RS
WY IP PKG AMCR SEE AVY BLL CCK SON
DAL UAL AAL LUV JBLU ALK
MAR HLT MGM LVS WYNN
MCO SPGI ICE CME NDAQ MKTX
DHI LEN PHM NVR TOL KBH
BX KKR APO ARES CG
AIG PRU MET AFL PFG LNC GL ALL TRV CB AXP PGR HIG CINF WRB
PPG SHW NUE DD DOW LYB EMN CE APD ECL
KMB CLX CHD PG CL EL COTY
""".split()

args = get_args()

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
