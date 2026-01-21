"""
Annual Report Parser Script.

Uses LLM to extract financial information from annual reports (PDF format).
The script:

- Loads annual report PDFs
- Extracts relevant pages containing financial data
- Uses LLM to parse and structure financial ratios and metrics
- Outputs structured data for downstream credit rating prediction

This standalone script focuses solely on document parsing and information
extraction, supporting the broader credit analysis pipeline.
"""

from jpm.config import Config, DataConfig, LLMConfig, get_args
from jpm.question_1 import LLMClient
from jpm.utils import set_seed

if __name__ == "__main__":
    set_seed(42)
    args = get_args()

    data_cfg = DataConfig.from_args(args)

    # Required for parsing reports
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o-2024-08-06",
    )

    config = Config(llm=llm_config, data=data_cfg)

    client = LLMClient()
    data = client.parse_annual_report(config)
