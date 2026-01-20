def get_ticker_prompt(company_names: list[str]) -> list[dict[str, str]]:
    system_content = (
        "You are a financial data assistant. When given a list of company names, "
        "return a JSON mapping of each company name to its official stock ticker "
        "symbol. Return ONLY valid JSON with no additional text, explanation, or "
        "markdown formatting."
    )

    # Format as numbered list
    company_list = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(company_names))

    user_content = (
        f"Map these company names to their official stock tickers:\n\n"
        f"{company_list}\n\n"
        "Return ONLY a JSON object in this exact format:\n"
        "{\n"
        '  "Company Name 1": "TICKER1",\n'
        '  "Company Name 2": "TICKER2"\n'
        "}\n"
        "Use null if ticker cannot be determined. Use the exact company names as keys."
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages


def get_company_name_prompt(tickers: list[str]) -> list[dict[str, str]]:
    system_content = (
        "You are a financial data assistant. When given a list of stock ticker "
        "symbols, return a JSON mapping of each ticker to a list of all common "
        "names and variations that company is known by (legal name, trading name, "
        "common abbreviations, etc.). Return ONLY valid JSON with no additional "
        "text, explanation, or markdown formatting."
    )

    # Format as numbered list
    ticker_list = "\n".join(f"{i + 1}. {ticker}" for i, ticker in enumerate(tickers))

    user_content = (
        f"Map these stock tickers to all possible company names and variations:\n\n"
        f"{ticker_list}\n\n"
        "Return ONLY a JSON object in this exact format:\n"
        "{\n"
        '  "TICKER1": ["Full Legal Name", "Common Name", "Abbreviation", ...],\n'
        '  "TICKER2": ["Full Legal Name", "Common Name", "Abbreviation", ...]\n'
        "}\n"
        "Include: legal name, trading name, common abbreviations, \
            and well-known variants. "
        "Use null if ticker is invalid. Use the exact ticker symbols as keys."
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages
