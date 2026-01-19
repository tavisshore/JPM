from jpm.question_1 import Config, DataConfig, LLMClient, LLMConfig, get_args, set_seed

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
