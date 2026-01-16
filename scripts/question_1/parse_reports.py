import argparse

from jpm.question_1 import LLMClient, LLMConfig

# Simple dict of the reports in the question
# Reduces token API usage
input_reports = {
    "alibaba": {"path": "assets/question_1/alibaba_report.pdf", "pages": (254, 264)},
    "exxon": {"path": "assets/question_1/exxon_report.pdf", "pages": (88, 92)},
    "gm": {"path": "assets/question_1/gm_report.pdf", "pages": (61, 64)},
    "goog": {"path": "assets/question_1/goog_report.pdf", "pages": (57, 61)},
    "jpm": {
        "path": "assets/question_1/jpm_report.pdf",
        "pages": (0, -1),
    },  # Narrow down
    "lvmh": {"path": "assets/question_1/lvmh_report.pdf", "pages": (24, 28)},
    "msft": {"path": "assets/question_1/msft_report.pdf", "pages": (44, 48)},
    "tencent": {"path": "assets/question_1/tencent_report.pdf", "pages": (124, 134)},
    "vw": {"path": "assets/question_1/vw_report.pdf", "pages": (472, 478)},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse annual reports using LLMClient")
    parser.add_argument(
        "--company",
        type=str,
        choices=input_reports.keys(),
        required=True,
        help="Company whose report to parse",
    )
    args = parser.parse_args()

    client = LLMClient()
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-5-mini",
    )

    report = input_reports[args.company]

    data = client.parse_annual_report(
        pdf_path=report["path"],
        cfg=llm_config,
        page_range=report["pages"],
    )
