def get_report_prompt(pdf_text: str) -> str:
    extraction_prompt = (
        "Extract these EXACT values from the most recent complete fiscal year "
        "statements:\n\n"
        "STEP 1: IDENTIFY FISCAL YEAR END DATE\n"
        "- Look for: 'as of [DATE]', 'for the year ended [DATE]', "
        "'December 31, [YEAR]'\n"
        "- Extract the most recent complete fiscal year end date\n"
        "- CRITICAL: Convert to ISO format YYYY-MM-DD\n"
        "  Examples:\n"
        "  - 'December 31, 2024' -> '2024-12-31'\n"
        "  - '31 Dec 2024' -> '2024-12-31'\n"
        "  - '31/12/2024' -> '2024-12-31'\n\n"
        "STEP 2: EXTRACT VALUES (in millions, original currency)\n"
        "From Balance Sheet for the identified fiscal year:\n"
        "1. Current assets (total current assets line)\n"
        "2. Inventories (inventory or stocks line)\n"
        "   - If not explicitly stated or immaterial, use null\n"
        "3. Current liabilities (total current liabilities line)\n"
        "4. Financial liabilities - non-current:\n"
        "   - IFRS: 'Non-current financial liabilities' or 'Borrowings - non-current'\n"
        "   - US GAAP: 'Long-term debt' or 'Long-term debt, \
            excluding current portion'\n"
        "   - Include: Finance lease liabilities (non-current)\n"
        "   - If multiple line items exist (debt + leases), sum them\n"
        "5. Financial liabilities - current:\n"
        "   - IFRS: 'Current financial liabilities' or 'Borrowings - current'\n"
        "   - US GAAP: 'Current portion of long-term debt' or \
              'Short-term debt' or 'Commercial paper'\n"
        "   - Include: Current portion of finance leases\n"
        "   - If multiple line items exist, sum them\n"
        "6. Equity (total equity line)\n"
        "7. Total assets\n\n"
        "From Income Statement for the same fiscal year:\n"
        "8. Net income (Earnings after tax / Net income / Profit for the year)\n"
        "9. EBIT (Operating result / Operating profit / Operating income)\n"
        "10. Interest expense:\n"
        "    - Look for: 'Interest expense' or 'Finance costs' or \
            'Interest and other finance costs'\n"
        "    - If not explicitly stated, calculate as: EBIT \
        - EBT (Earnings before tax)\n"
        "    - If cannot be found or calculated, use null\n"
        "11. Revenue (Sales revenue / Revenue / Total revenue)\n"
        "12. Operating expenses (sum of Distribution + Administrative + Other "
        "operating expenses)\n\n"
        "From Cash Flow Statement or Notes:\n"
        "13. Depreciation and amortization:\n"
        "    - Look for: 'Depreciation and amortization' line in cash flow statement\n"
        "    - Or in notes: sum of 'Depreciation' + 'Amortization'\n"
        "    - If not explicitly stated, calculate as: EBITDA - \
              EBIT (if EBITDA is available)\n"
        "    - If cannot be found or calculated, use null\n\n"
        "STEP 3: IDENTIFY CURRENCY\n"
        "- Look for currency indicator in report "
        "(€ = EUR, $ = USD, ¥ = JPY, £ = GBP, ¥ = CNY)\n"
        "- Extract the 3-letter ISO currency code\n\n"
        "CRITICAL RULES:\n"
        "- Extract values from the SAME fiscal year for all items\n"
        "- Use the MOST RECENT complete fiscal year available\n"
        "- Values must be in millions (original currency)\n"
        "- Extract EXACTLY as printed when available\n"
        "- For calculated fields (interest, D&A), \
        only calculate if source data exists\n"
        "- Use null (not 0) when a value is not available or not applicable\n"
        "- fiscal_year_end must be the exact date from the statements in "
        "YYYY-MM-DD format\n\n"
        "Return JSON with this exact structure:\n"
        "{\n"
        '  "current_assets": <number>,\n'
        '  "inventories": <number or null>,\n'
        '  "current_liabilities": <number>,\n'
        '  "financial_liabilities_noncurrent": <number or null>,\n'
        '  "financial_liabilities_current": <number or null>,\n'
        '  "equity": <number>,\n'
        '  "total_assets": <number>,\n'
        '  "net_income": <number>,\n'
        '  "ebit": <number>,\n'
        '  "interest_expense": <number or null>,\n'
        '  "revenue": <number>,\n'
        '  "operating_expenses": <number>,\n'
        '  "depreciation_amortization": <number or null>,\n'
        '  "currency": "<ISO code like EUR, USD, JPY, CNY, GBP>",\n'
        '  "fiscal_year_end": "<YYYY-MM-DD>"\n'
        "}\n\n"
        "Numbers must have NO commas. All values must be plain numbers or null.\n\n"
        f"{pdf_text}"
    )
    return extraction_prompt
