from __future__ import annotations

import random

import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    """Set seeds across Python, NumPy, and TensorFlow for reproducibility."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_tickers(sector="tech", length=-1):
    # Technology
    tech = """
    AAPL MSFT GOOGL AMZN NVDA META TSLA ORCL ADBE CRM INTC AMD QCOM AVGO CSCO
    INTU NOW AMAT LRCX KLAC SNPS CDNS MCHP NXPI ADSK FTNT PANW CRWD ZS DDOG NET
    """.split()

    # Banks & Financial Services
    banks = """
    JPM BAC WFC C GS MS USB PNC BK TFC SCHW BLK STT NTRS RF CFG KEY HBAN FITB ZION
    CMA MTB SIVB FRC WAL EWBC SBNY PB ONB
    """.split()

    # Energy
    energy = """
    XOM CVX COP SLB EOG MPC PSX VLO HAL OXY DVN FANG HES MRO APA BKR NOV FTI
    KMI WMB OKE EPD TRGP ET MMP PAA
    """.split()

    # Healthcare & Pharma
    healthcare = """
    JNJ PFE MRK ABBV LLY BMY GILD AMGN REGN VRTX BIIB ZTS ISRG BSX MDT ABT TMO DHR
    SYK BDX BAX HOLX DXCM ALGN IDXX IQV PKI WAT MTD A PODD TECH RMD
    """.split()

    # Retail & Consumer Discretionary
    retail = """
    WMT TGT HD LOW COST CVS WBA DG DLTR ROST TJX BBY EBAY ETSY W CHWY BURL FIVE
    GPS ANF AEO URBN DKS BBWI
    """.split()

    # Consumer Staples & Food/Beverage
    consumer_staples = """
    KO PEP MDLZ GIS K MO PM STZ TAP BF.B CAG CPB SJM HSY KR SYY ADM BG TSN HRL
    INGR KMB CLX CHD PG CL EL COTY KHC MNST DPZ YUM MCD SBUX CMG QSR
    """.split()

    # Media & Telecom
    media_telecom = """
    T VZ TMUS CMCSA CHTR DIS NFLX PARA FOX FOXA WBD DISH LUMN SIRI WMG OMC IPG
    """.split()

    # Aerospace & Defense
    aerospace_defense = """
    BA RTX LMT GD NOC HON TDG HWM LDOS AJRD HEI TXT SPR WWD AVAV KTOS
    """.split()

    # Industrials & Machinery
    industrials = """
    CAT DE MMM ETN EMR ITW ROK DOV PH IR XYL FTV AME GNRC IEX CR CARR OTIS TT
    JCI BLDR VMC MLM SUM CRH FAST GWW MSM WSO WESCO DCI AIT
    """.split()

    # Healthcare Insurance
    health_insurance = """
    UNH CI HUM ANTM CNC CVS ELV MOH
    """.split()

    # Payment Processors & Credit Cards
    payments = """
    MA V AXP DFS COF SYF
    """.split()

    # Automotive
    automotive = """
    F GM TSLA STLA TM HMC RIVN LCID
    """.split()

    # Transportation & Logistics
    transportation = """
    UPS FDX XPO JBLU CHRW EXPD LSTR ODFL SAIA ARCB
    """.split()

    # REITs
    reits = """
    AMT CCI EQIX PLD SPG O VICI AVB EQR MAA ESS UDR CPT BXP VTR WELL PEAK HST
    RHP SLG KRC DEI ARE DLR QTS CONE COR FR BRX KRG REG WPC NNN STAG REXR EGP
    """.split()

    # Utilities
    utilities = """
    SO DUK NEE D AEP EXC ED SRE XEL PCG WEC ES PEG ETR FE CNP CMS NI DTE AEE
    LNT EVRG ATO NWE PNW AVA OGE
    """.split()

    # Materials & Mining
    materials = """
    NEM FCX SCCO GOLD AA ALB MP FMC CF MOS APD LIN ECL NUE STLD CLF X RS CENX
    CMC MT TECK VALE RIO BHP
    """.split()

    # Packaging & Paper
    packaging = """
    WY IP PKG AMCR SEE AVY BLL CCK SON GPK
    """.split()

    # Airlines
    airlines = """
    DAL UAL AAL LUV JBLU ALK SAVE HA
    """.split()

    # Hospitality & Gaming
    hospitality = """
    MAR HLT MGM LVS WYNN CZR PENN BYD CHDN RCL CCL NCLH
    """.split()

    # Financial Exchanges & Data
    exchanges = """
    MCO SPGI ICE CME NDAQ MKTX CBOE
    """.split()

    # Homebuilders
    homebuilders = """
    DHI LEN PHM NVR TOL KBH MTH TMHC MHO LGIH TPH BZH GRBK
    """.split()

    # Private Equity & Asset Management
    private_equity = """
    BX KKR APO ARES CG BLUE OWL STEP HTGC ARCC
    """.split()

    # Insurance
    insurance = """
    AIG PRU MET AFL PFG LNC GL ALL TRV CB AXP PGR HIG CINF WRB RLI KMPR SIGI AFG
    ORI EG JRVR THG
    """.split()

    # Chemicals & Coatings
    chemicals = """
    PPG SHW NUE DD DOW LYB EMN CE APD ECL ALB FMC RPM AXTA VVV HUN WLK OLN CC
    """.split()

    # Combine all into single list
    sector_map = {
        "tech": tech,
        "banks": banks,
        "energy": energy,
        "healthcare": healthcare,
        "retail": retail,
        "consumer_staples": consumer_staples,
        "media_telecom": media_telecom,
        "aerospace_defense": aerospace_defense,
        "industrials": industrials,
        "health_insurance": health_insurance,
        "payments": payments,
        "automotive": automotive,
        "transportation": transportation,
        "reits": reits,
        "utilities": utilities,
        "materials": materials,
        "packaging": packaging,
        "airlines": airlines,
        "hospitality": hospitality,
        "exchanges": exchanges,
        "homebuilders": homebuilders,
        "private_equity": private_equity,
        "insurance": insurance,
        "chemicals": chemicals,
        "all": tech
        + banks
        + energy
        + healthcare
        + retail
        + consumer_staples
        + media_telecom
        + aerospace_defense
        + industrials
        + health_insurance
        + payments
        + automotive
        + transportation
        + reits
        + utilities
        + materials
        + packaging
        + airlines
        + hospitality
        + exchanges
        + homebuilders
        + private_equity
        + insurance
        + chemicals,
    }
    data_tickers = sector_map.get(sector, [])
    data_tickers = list(set(data_tickers))
    if length > 0:
        data_tickers = data_tickers[:length]
    return data_tickers


def compute_results_summary(results, tickers, config_name):
    print(f"\n--- Results Summary for: {config_name} ---")
    summary = {}

    # Get list of tickers that actually have results
    valid_tickers = [ticker for ticker in tickers if ticker in results]

    if not valid_tickers:
        print("No valid tickers found in results")
        return summary, valid_tickers

    for key in ["net_income", "balance_sheet"]:
        # Get model names from first valid ticker
        if key in results[valid_tickers[0]]:
            for model_name in results[valid_tickers[0]][key].keys():
                summary_key = f"{key}_{model_name}"
                # Only use tickers that have this key and model
                vals = [
                    results[ticker][key][model_name]
                    for ticker in valid_tickers
                    if key in results[ticker] and model_name in results[ticker][key]
                ]
                if vals:
                    mean_val = np.mean(vals)
                    std_val = np.std(vals)
                    summary[summary_key] = {"mean": mean_val, "std": std_val}

    return summary, valid_tickers
