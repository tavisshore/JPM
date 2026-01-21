import argparse
from pathlib import Path


def _print_category_help(category: str):
    """Print help for a specific config category."""
    help_text = {
        "data": """
DataConfig Arguments (Data Loading and Preprocessing):
  --ticker TICKER               Stock ticker symbol (default: AAPL)
  --industry INDUSTRY           Industry sector filter, available:
                                all, tech, banks, energy, healthcare,
                                retail, consumer_staples, media_telecom,
                                aerospace_defense, industrials, health_insurance,
                                payments, automotive, transportation, reits,
                                utilities, materials, packaging, airlines,
                                hospitality, exchanges, homebuilders,
                                private_equity, insurance, chemicals
  --cache-dir CACHE_DIR         Directory for caching downloaded data
  --save-dir SAVE_DIR           Directory for saving outputs
  --plots-dir PLOTS_DIR         Directory for saving plots
  --periods PERIODS             Number of quarters to include (default: 60)
  --lookback LOOKBACK           Number of historical quarters for input (default: 4)
  --horizon HORIZON             Number of future quarters to predict (default: 1)
  --batch-size BATCH_SIZE       Batch size for training (default: 32)
  --target-type {full,bs,net_income}
                                Type of target variable (default: full)
  --withhold-periods WITHHOLD   Number of quarters to withhold for testing (default: 1)
  --seasonal-weight WEIGHT      Weight applied to seasonal lag (default: 1.1)
""",
        "lstm": """
LSTMConfig Arguments (Model Architecture and Training):

Architecture:
  --lstm-units UNITS            Number of units per LSTM layer (default: 256)
  --lstm-layers LAYERS          Number of stacked LSTM layers (default: 2)
  --hidden-units UNITS          Alias for lstm-units
  --dense-units UNITS           Number of units in dense layers (default: 256)
  --dropout RATE                Dropout rate for regularization (default: 0.1)

Uncertainty Quantification (mutually exclusive):
  --probabilistic               Use probabilistic outputs (multivariate Gaussian)
  --variational                 Use variational inference (Bayesian weights)
  --mc-samples N                Monte Carlo samples for uncertainty (default: 1)

Training Hyperparameters:
  --epochs N                    Number of training epochs (default: 500)
  --lr RATE                     Learning rate (default: 1e-4)
  --decay-steps N               Steps for LR decay (default: 100)
  --decay-rate RATE             LR scheduler decay rate (default: 0.9)
  --scheduler {exponential,cosine,constant}
                                Type of LR scheduler (default: exponential)
  --checkpoint-path PATH        Path to save checkpoints (default: ckpts)

Balance Sheet Constraints:
  --enforce-balance             Enforce BS identity: Assets = Liabilities + Equity
  --learn-identity              Learn BS identity as auxiliary loss
  --identity-weight WEIGHT      Weight for identity loss (default: 1e-4)
  --learn-subtotals             Learn subcategory subtotals as auxiliary loss
  --subcategory-weight WEIGHT   Weight for subcategory loss (default: 1e-5)
""",
        "llm": """
LLMConfig Arguments (Large Language Model Integration):
  --use-llm                     Enable LLM for forecast adjustment/generation
  --llm-provider PROVIDER       LLM provider name (default: openai)
  --llm-model MODEL             Model identifier (default: gpt-5-mini)
  --llm-temperature TEMP        Sampling temperature (default: 0.0)
  --llm-max-tokens N            Maximum tokens to generate (default: 8192)
  --adjust                      Use LLM to adjust LSTM predictions
                                (vs. independent generation then averaging)
""",
    }
    print(help_text[category])
    print("\nFor full help: python <script> --help")
    print("For other categories: --help {data,lstm,llm}")


def get_args():
    """Build CLI args for training entrypoints with all config options.

    Special help commands:
        --help data  : Show only DataConfig arguments
        --help lstm  : Show only LSTMConfig arguments
        --help llm   : Show only LLMConfig arguments
    """
    # Check for special help commands first
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--help":
        category = sys.argv[2].lower()
        if category in ["data", "lstm", "llm"]:
            _print_category_help(category)
            sys.exit(0)

    p = argparse.ArgumentParser(
        description="Financial Statement Forecasting with LSTM. Use '--help data', '--help lstm', or '--help llm' for category-specific help."
    )

    # DataConfig fields
    p.add_argument(
        "--ticker", type=str, default="MSFT", help="Stock ticker symbol (default: MSFT)"
    )
    p.add_argument(
        "--industry",
        type=str,
        default="all",
        help="Industry sector filter, available: all, tech, banks, energy, healthcare, retail, consumer_staples, media_telecom, aerospace_defense, industrials, health_insurance, payments, automotive, transportation, reits, utilities, materials, packaging, airlines, hospitality, exchanges, homebuilders, private_equity, insurance, chemicals",
    )
    p.add_argument(
        "--total-tickers",
        type=int,
        default=-1,
        help="Total number of tickers to process (default: -1)",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default="assets/question_1/",
        help="Directory for caching downloaded data",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default="results/question_1",
        help="Directory for saving outputs",
    )
    p.add_argument(
        "--plots-dir",
        type=str,
        default="results/question_1/plots",
        help="Directory for saving plots",
    )
    p.add_argument(
        "--periods",
        type=int,
        default=None,
        help="Number of quarters to include (default: 60)",
    )
    p.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Number of historical quarters for input sequences (default: 4)",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Number of future quarters to predict (default: 1)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training (default: 32)",
    )
    p.add_argument(
        "--target-type",
        type=str,
        default=None,
        choices=["full", "bs", "net_income"],
        help="Type of target variable (default: full)",
    )
    p.add_argument(
        "--withhold-periods",
        type=int,
        default=None,
        help="Number of quarters to withhold for testing (default: 1)",
    )
    p.add_argument(
        "--seasonal-weight",
        type=float,
        default=None,
        help="Weight applied to seasonal lag timestep (default: 1.1)",
    )

    # LSTMConfig fields - Architecture
    p.add_argument(
        "--lstm-units",
        type=int,
        default=None,
        help="Number of units in each LSTM layer (default: 256)",
    )
    p.add_argument(
        "--lstm-layers",
        type=int,
        default=None,
        help="Number of stacked LSTM layers (default: 2)",
    )
    p.add_argument(
        "--hidden-units", type=int, default=None, help="Alias for lstm_units"
    )
    p.add_argument(
        "--dense-units",
        type=int,
        default=None,
        help="Number of units in dense layers (default: 256)",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout rate for regularization (default: 0.1)",
    )

    # LSTMConfig fields - Uncertainty quantification (mutually exclusive)
    p.add_argument(
        "--probabilistic",
        action="store_true",
        help="Use probabilistic outputs with multivariate Gaussian distribution",
    )
    p.add_argument(
        "--variational",
        action="store_true",
        help="Use variational inference with Bayesian weights",
    )
    p.add_argument(
        "--mc-samples",
        type=int,
        default=None,
        help="Number of Monte Carlo samples for uncertainty estimation (default: 1)",
    )

    # LSTMConfig fields - Training hyperparameters
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 500)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for optimizer (default: 1e-4)",
    )
    p.add_argument(
        "--decay-steps",
        type=int,
        default=None,
        help="Number of steps for learning rate decay (default: 100)",
    )
    p.add_argument(
        "--decay-rate",
        type=float,
        default=None,
        help="Decay rate for learning rate scheduler (default: 0.9)",
    )
    p.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["exponential", "cosine", "constant"],
        help="Type of learning rate scheduler (default: exponential)",
    )
    p.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("ckpts"),
        help="Path to save model checkpoints",
    )

    # LSTMConfig fields - Balance sheet constraints and loss configuration
    p.add_argument(
        "--enforce-balance",
        action="store_true",
        help="Enforce balance sheet identity: Assets = Liabilities + Equity",
    )
    p.add_argument(
        "--learn-identity",
        action="store_true",
        help="Learn balance sheet identity as auxiliary loss",
    )
    p.add_argument(
        "--identity-weight",
        type=float,
        default=None,
        help="Weight for identity loss term (default: 1e-4)",
    )
    p.add_argument(
        "--learn-subtotals",
        action="store_true",
        help="Learn subcategory subtotals as auxiliary loss",
    )
    p.add_argument(
        "--subcategory-weight",
        type=float,
        default=None,
        help="Weight for subcategory loss term (default: 1e-5)",
    )
    p.add_argument(
        "--lambda-balance",
        type=float,
        default=None,
        help="Deprecated: use identity-weight instead",
    )

    # LLMConfig fields
    p.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for forecast adjustment or generation",
    )
    p.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        help="LLM provider name (default: openai)",
    )
    p.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Model identifier (default: gpt-5-mini)",
    )
    p.add_argument(
        "--llm-temperature",
        type=float,
        default=None,
        help="Sampling temperature for generation (default: 0.0)",
    )
    p.add_argument(
        "--llm-max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (default: 8192)",
    )
    p.add_argument(
        "--adjust",
        action="store_true",
        help="Use LLM to adjust LSTM predictions (vs. independent generation)",
    )

    args = p.parse_args()

    # Resolve cache_dir: CLI arg > env var > hardcoded default
    # if args.cache_dir is None:
    #     args.cache_dir = os.getenv("JPM_CACHE_DIR", "/assets/jpm")

    # Handle hidden_units as alias for lstm_units
    if args.hidden_units is not None and args.lstm_units is None:
        args.lstm_units = args.hidden_units

    return args
