import json

from jpm.question_1.data.structures import get_fs_struct


def get_statement_prompt(statement_type, features):
    fs_dictionary = get_fs_struct(statement_type)
    structure_json = fs_dictionary["prediction_structure"]
    mapping_examples = fs_dictionary["mapping_examples"]
    guidelines = fs_dictionary["classification_guidelines"]

    structure_str = json.dumps(
        structure_json, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    features_str = json.dumps(features, ensure_ascii=False)

    examples_context = "\n".join(
        [
            f"- {category}: {', '.join(examples[:5])}"
            for category, examples in mapping_examples.items()
        ]
    )

    general_rules = "\n".join([f"  - {rule}" for rule in guidelines["General Rules"]])
    specific_mappings = "\n".join(
        [
            f"  - {category}: {desc}"
            for category, desc in guidelines["Specific Mappings"].items()
        ]
    )
    edge_cases = "\n".join(
        [
            f"  - {case}: {desc}"
            for case, desc in guidelines.get("Edge Cases", {}).items()
        ]
    )

    rollup_detection = ""
    if "Rollup Detection" in guidelines:
        rollup_rules = guidelines["Rollup Detection"]
        if isinstance(rollup_rules, dict):
            lines = []
            for category, rules in rollup_rules.items():
                lines.append(f"  {category}:")
                for rule in rules:
                    lines.append(f"    - {rule}")
            rollup_detection = "\n".join(lines)
        elif isinstance(rollup_rules, list):
            rollup_detection = "\n".join([f"  - {rule}" for rule in rollup_rules])

    keyword_rules = ""
    if statement_type == "income_statement":
        keyword_rules = """
        CRITICAL KEYWORD MATCHING RULES FOR INCOME STATEMENT (APPLY WITH CONFIDENCE):

        ANTI-ROLLUP EXCEPTIONS (HIGHEST PRIORITY - CHECK FIRST):
        The following fields are NOT rollups and must ALWAYS be mapped:
        1. Exact field "revenues" -> Total Revenues
        2. Exact field "operating expenses" -> Total Operating Expenses
        3. "revenue from contract with customer" -> Total Revenues

        These fields may coexist with their components - this is NORMAL for validation.
        Map BOTH the total AND the components when they exist.

        If a feature contains these keywords, map it regardless of verbose phrasing:

        1. REVENUE FIELDS (map ALL - never rollups):
            a. Exact "revenues" OR exact "revenue"
            -> Total Revenues
            b. "revenue from contract with customer"
            -> Total Revenues
            c. "net sales"
            -> Total Revenues

            CRITICAL: If you see both "revenues" AND "revenue from contract with customer",
            they are ALTERNATE TAXONOMIES (mutually exclusive by period).
            Map BOTH - they are NOT a parent/child rollup.

        2. OPERATING EXPENSES (map the total even if components exist):
            a. Exact "operating expenses" OR "total operating expenses"
            -> Total Operating Expenses
            [NOT __unmapped__ - even if SG&A and R&D exist separately]
            b. "selling general and administrative"
            -> General and Administrative
            c. "research and development"
            -> Research and Development
            d. "sales and marketing"
            -> Sales and Marketing
            e. "fulfillment"
            -> Other Operating Expenses

            CRITICAL: It is NORMAL and EXPECTED for both "operating expenses" (total)
            and its components (SG&A, R&D) to exist. Map ALL of them for validation.

        3. INCOME BEFORE TAXES (prefer simpler versions):
            a. "before income taxes" OR "before taxes" OR "pretax"
            -> Income Before Taxes
            b. If multiple variations exist, prefer the SHORTEST field name
            Example: "income before taxes" > "income from continuing operations before taxes..."

        4. NON-OPERATING ITEMS:
            a. "nonoperating income expense"
            -> Total Non-Operating Items
            b. "interest expense"
            -> Interest Expense
            c. "interest income"
            -> Interest Income
            d. "other income" OR "other expense" (without "comprehensive")
            -> Other Income (Expense)

        5. COST OF REVENUE:
            a. "cost of goods and services sold" OR "cost of revenue" OR "cost of sales"
            -> Total Cost of Revenue

        6. ROLLUP DETECTION (only apply to these specific patterns):
            ONLY mark as __unmapped__ if the field name contains:
            a. "costs and expenses" (combines COGS + OpEx)
            -> __unmapped__
            b. Multiple concepts joined with "and" that are distinct P&L categories
            (e.g., "revenue and other income" if both exist separately)
            -> __unmapped__

            DO NOT mark as rollups:
            - "operating expenses" (this is a required total)
            - "revenues" (this is a required total)
            - Any field matching sections 1-5 above

        7. METADATA (always unmapped):
            a. "dividends per share"
            -> __unmapped__

        DECISION FRAMEWORK:
        For each field, check in this order:
        1. Is it in ANTI-ROLLUP EXCEPTIONS? → Map immediately, skip rollup check
        2. Does it match sections 1-5? → Map it
        3. Does it match section 6 rollup patterns? → Unmapped
        4. Is it metadata (section 7)? → Unmapped
        5. Otherwise → Map to best matching category

        These patterns override ALL other considerations including rollup detection.
        """
    elif statement_type == "balance_sheet":
        keyword_rules = """
        CRITICAL KEYWORD MATCHING RULES FOR BALANCE SHEET (APPLY WITH CONFIDENCE):
        If a feature contains these keywords, map it regardless of verbose phrasing:

        1. TOTAL MAPPINGS (highest priority - map these FIRST):
            a. Exact "assets" (not "assets current" or "assets noncurrent")
                -> Total Assets
            b. "assets current" OR "current assets" OR "total current assets"
                -> Total Current Assets
            c. "assets noncurrent" OR "noncurrent assets" OR "non current assets"
                OR "total non-current assets" OR "total noncurrent assets"
                -> Total Non-Current Assets
            d. Exact "liabilities" (not "liabilities current" or "liabilities noncurrent")
                -> Total Liabilities
            e. "liabilities current" OR "current liabilities" OR "total current liabilities"
                -> Total Current Liabilities
            f. "liabilities noncurrent" OR "noncurrent liabilities" OR "non current liabilities"
                OR "total non-current liabilities" OR "total noncurrent liabilities"
                -> Total Non-Current Liabilities
            g. "stockholders equity" OR "shareholders equity" OR "stockholder equity"
                OR "shareholder equity" OR "total equity"
                -> Total Equity
            h. "liabilities and stockholders equity" OR "liabilities and equity"
                OR "total liabilities and equity"
                -> Total Liabilities and Equity

        2. MARKETABLE SECURITIES - ALTERNATE TAXONOMIES (map ALL variations):
            CRITICAL: These fields represent the SAME economic concept reported under
            different GAAP tags. They are MUTUALLY EXCLUSIVE by period (never both
            non-zero in same quarter). Map ALL of them - they are NOT rollups.

            a. "available for sale securities current" -> Marketable Securities (Short-term)
            b. "marketable securities current" -> Marketable Securities (Short-term)
            c. "short term investments" OR "short-term investments"
                -> Marketable Securities (Short-term)
            d. "commercial paper" -> Marketable Securities (Short-term)
            e. "available for sale securities noncurrent" -> Long-term Investments
            f. "marketable securities noncurrent" -> Long-term Investments

            These are EQUIVALENT reporting methods, NOT parent/child relationships.
            Do NOT apply rollup detection to these fields.

        3. DEFERRED TAX MAPPINGS (handle net positions):
            a. "deferred tax asset" (without "net" or "liability")
                -> Deferred Tax Assets
            b. "deferred tax liabilit" (without "net" or "asset")
                -> Deferred Tax Liabilities
            c. "deferred tax assets liabilities net" OR "deferred income taxes net"
                -> Deferred Tax Assets

            SPECIAL HANDLING: Net deferred tax positions will be split by sign
            in preprocessing - always map to Deferred Tax Assets

        4. LEASE LIABILITIES (critical for liability completeness):
            a. "operating lease liability" OR "operating lease liabilities"
                -> Lease Liabilities
            b. "finance lease liability" OR "finance lease liabilities"
                -> Lease Liabilities

        5. TREASURY STOCK (critical for equity calculation):
            a. "treasury stock" OR "treasury shares"
                -> Treasury Stock

        6. METADATA/NON-FINANCIAL (always unmapped):
            a. "shares issued" OR "shares outstanding" OR "shares authorized"
                -> __unmapped__ (share counts, not dollar values)
            b. "par value" (when standalone, not in "common stock par value")
                -> __unmapped__ (metadata)

        7. ROLLUP DETECTION (prevents double-counting):
            EXCEPTION: Do NOT apply rollup detection to marketable securities fields
            (section 2 above) - those are alternate taxonomies.

            a. Features combining multiple concepts in their name
                (e.g., "cash cash equivalents and short term investments")
                -> __unmapped__
            b. If you see both a specific component (e.g., "prepaid expenses")
                AND a clearly labeled aggregate (e.g., "prepaid and other current assets"),
                prefer the aggregate
            c. True rollups have names that explicitly combine multiple concepts
                (look for "and" in the field name combining distinct items)

        PRIORITY ORDER:
        1. Apply TOTAL MAPPINGS first (these are never rollups)
        2. Apply MARKETABLE SECURITIES alternate taxonomy mappings (NOT rollups)
        3. Apply DEFERRED TAX, LEASE, TREASURY STOCK mappings
        4. Check for TRUE rollups (combined field names with "and")
        5. Map remaining features to specific line items
        6. Place unmappable items in __unmapped__

        CRITICAL ANTI-DOUBLE-COUNTING RULES:
        - "available for sale securities" and "marketable securities" are ALTERNATE
            TAXONOMIES (same thing, different GAAP tags) - map BOTH, never __unmapped__
        - Only apply rollup detection when a field name EXPLICITLY combines concepts
            (e.g., "cash and investments" would be a rollup if both "cash" and
            "investments" exist separately)
        - When in doubt about securities fields: MAP THEM (they're mutually exclusive)

        These patterns override general caution - apply them confidently.
        """
    elif statement_type == "cash_flow":
        keyword_rules = """
        CRITICAL KEYWORD MATCHING RULES FOR CASH FLOW (APPLY WITH CONFIDENCE):
        If a feature contains these keywords, map it regardless of verbose phrasing:

        1. NET CASH FLOW TOTALS (highest priority - map these FIRST):
            a. "net cash provided by operating" OR "net cash from operating"
            -> Net Cash from Operating Activities
            b. "net cash used in investing" OR "net cash from investing"
            -> Net Cash from Investing Activities
            c. "net cash provided by financing" OR "net cash from financing"
            -> Net Cash from Financing Activities
            d. "cash" AND "period increase decrease"
            -> Net Change in Cash
            e. "cash equivalents" AND "increase decrease including exchange rate"
            -> Net Change in Cash

        2. CASH POSITION FIELDS:
            a. "cash" AND "beginning of period"
            -> Cash at Beginning of Period
            b. "cash" AND "end of period"
            -> Cash at End of Period
            c. "effect of exchange rate" OR "exchange rate effect" OR "exchange rate changes"
            -> Effect of Exchange Rate Changes

        3. OPERATING ACTIVITIES:
            a. "changes in working capital" OR "changes in operating assets and liabilities"
            -> Changes in Operating Assets and Liabilities
            b. "depreciation" OR "amortization" OR "depreciation and amortization"
            -> Depreciation and Amortization
            c. "deferred income tax" OR "deferred tax"
            -> Deferred Income Taxes
            d. "share based compensation" OR "stock based compensation"
            -> Stock-Based Compensation

        4. INVESTING ACTIVITIES:
            a. "acquisitions net of cash acquired"
            (with or without additional text like "and purchases of...")
            -> Acquisitions (net of cash acquired)
            b. "payments to acquire property plant and equipment" OR "capital expenditures"
            -> Capital Expenditures
            c. "payments to acquire available for sale securities" OR "purchases of investments"
            -> Purchases of Investments
            d. "proceeds from sale" AND "securities"
            -> Sales and Maturities of Investments
            e. "proceeds from maturities" OR "proceeds from maturity"
            -> Sales and Maturities of Investments

        5. FINANCING ACTIVITIES:
            a. "payments of dividends" (with or without "and dividend equivalents")
            -> Dividends Paid
            b. "payments for repurchase" OR "stock repurchases"
            -> Stock Repurchases
            c. "proceeds from issuance of common stock"
            -> Proceeds from Stock Issuance
            d. "proceeds from issuance of" AND "debt"
            -> Proceeds from Debt
            e. "repayments of" AND "debt"
            -> Repayment of Debt

        6. SUPPLEMENTAL DISCLOSURES:
            a. "income taxes paid" OR "cash paid for income taxes"
            -> Other Non-Cash Items
            [Supplemental disclosure - different from deferred tax expense]
            b. "interest paid"
            -> Interest Paid

        7. DIVIDEND VARIATIONS (alternate taxonomies):
            a. "payments of dividends" (simple version)
            -> Dividends Paid
            b. "payments of dividends and dividend equivalents on common stock"
            (detailed version including RSU dividend equivalents)
            -> Dividends Paid
            CRITICAL: These are ALTERNATE DETAIL LEVELS of the same cash outflow.
            Map BOTH if both exist - they are mutually exclusive by period.

        8. ROLLUP DETECTION (prevents double-counting):
            - Cash flow statements rarely have true rollups
            - Most "combined" fields are legitimate aggregations
            - Only unmapped if field explicitly says "subtotal" or "total"
            AND more granular components exist

        PRIORITY ORDER:
        1. Apply NET CASH FLOW TOTALS first (section 1)
        2. Apply CASH POSITION fields (section 2)
        3. Map activity-specific line items (sections 3-5)
        4. Map SUPPLEMENTAL DISCLOSURES (section 6)
        5. Handle DIVIDEND VARIATIONS (section 7)
        6. Check for true rollups (section 8 - rare in cash flow)

        ANTI-DOUBLE-COUNTING STRATEGY:
        - Dividend variations are ALTERNATE DETAIL LEVELS, not rollups
        - "income taxes paid" is supplemental (cash basis), separate from
        "deferred income tax" (accrual basis) - both should be mapped
        - Cash flow line items typically don't have parent/child relationships
        like balance sheet does

        These patterns override general caution - apply them confidently.
        """

    statement_label = statement_type.replace("_", " ")
    rollup_fallback = "  - Check for combined/rollup features that cause double-counting"  # noqa: E501

    system_content = f"""
      You are a deterministic mapper from feature strings to a fixed \
      {statement_label} JSON taxonomy.

      DEFINITIONS:
      - A "leaf" is any key whose value is a JSON array (list). Only leaves may be modified.
      - Non-leaves are JSON objects (dicts) and must not be modified.

      HARD REQUIREMENTS:
      - Output must be a single valid JSON object.
      - The output object MUST keep identical keys and nesting as the input structure
      (no new keys; no renamed keys).
      - Only append items to existing leaf arrays.
      - Feature strings must be copied verbatim (exact characters; no normalization).

      {keyword_rules}

      CRITICAL TOTAL/ROLLUP MAPPING RULES (APPLY FIRST):
      Balance Sheet totals MUST be mapped as follows:
      1. Features matching "assets" (exact) or "total assets"
         -> Assets / Total Assets
      2. Features matching "assets current" or "total current assets"
         -> Assets / Current / Total Current Assets
      3. Features matching "assets noncurrent" or "total non-current assets"
         or "assets non current" -> Assets / Non-Current / Total Non-Current Assets
      4. Features matching "liabilities" (exact) or "total liabilities"
         -> Liabilities / Total Liabilities
      5. Features matching "liabilities current" or "total current liabilities"
         -> Liabilities / Current / Total Current Liabilities
      6. Features matching "liabilities noncurrent" or "total non-current liabilities"
         -> Liabilities / Non-Current / Total Non-Current Liabilities
      7. Features matching "stockholders equity" or "shareholders equity"
         or "total equity" -> Equity / Total Equity
      8. Features matching "liabilities and stockholders equity" or similar
         -> Totals / Total Liabilities and Equity

      Income Statement totals:
      1. Features matching "revenue" or "total revenue" or "net sales"
         -> Revenues / Total Revenues
      2. Features matching "cost of revenue" or "cost of sales"
         -> Cost of Revenue / Total Cost of Revenue
      3. Features matching "operating expenses"
         -> Operating Expenses / Total Operating Expenses
      4. Features matching "net income" or "net earnings" -> Net Income

      Cash Flow totals:
      1. Features matching "net cash provided by operating activities"
         or "net cash from operating"
         -> Operating Activities / Net Cash from Operating Activities
      2. Features matching "net cash used in investing activities"
         or "net cash from investing"
         -> Investing Activities / Net Cash from Investing Activities
      3. Features matching "net cash provided by financing activities"
         or "net cash from financing"
         -> Financing Activities / Net Cash from Financing Activities

      ROLLUP DETECTION (CRITICAL - PREVENTS DOUBLE-COUNTING):
      {rollup_detection if rollup_detection else rollup_fallback}

      MAPPING GUIDELINES:
      {general_rules}

      SPECIFIC CATEGORY MAPPINGS:
      {specific_mappings}

      EDGE CASES:
      {edge_cases}

      MAPPING EXAMPLES (for reference, not exhaustive):
      {examples_context}

      PRIORITIZATION STRATEGY:
      1. FIRST: Check if feature matches CRITICAL KEYWORD MATCHING RULES above -
         apply those with confidence
      2. SECOND: Check if the feature is a ROLLUP that combines other separate
         features - if yes, map to __unmapped__ to prevent double-counting
      3. THIRD: Check if the feature matches any CRITICAL TOTAL/ROLLUP rules above -
         if yes, map there immediately
      4. FOURTH: Check if feature represents a meaningful financial statement line item
      5. IGNORE IRRELEVANT FEATURES: Skip features that are:
      - Meta-information (filing dates, document IDs, entity information, CIK numbers)
      - Ratios or percentages (unless specifically part of per-share data)
      - Share counts (unless in per-share data section)
      - Duplicative or redundant information
      - Administrative fields
      6. Place ignored features into "__unmapped__" array

      MAPPING PROCESS FOR NON-TOTAL FEATURES:
      1. For each feature, determine if it represents a valid line item for this
         statement type
      2. If valid, choose the single best leaf category by:
      a. Match to the most specific applicable category
      b. Use mapping examples as guidance (not strict rules)
      c. Prefer specific categories over "Other" categories
      d. If ambiguous, choose the category with the closest accounting meaning
      3. If invalid or unclear, place in "__unmapped__"

      QUALITY OVER COMPLETENESS:
      - It is BETTER to leave a feature unmapped than to force it into an incorrect
      category
      - An empty array for a line item is acceptable if no relevant feature exists
      - "__unmapped__" should contain meta-information, non-financial items, AND
      rollup features that would cause double-counting
      - However, features matching CRITICAL KEYWORD RULES should be mapped
      confidently, not conservatively

      SELF-CHECK (silent before output):
      - Total mapped items across ALL leaf lists (including __unmapped__) equals
      number of input features
      - No duplicates across leaves
      - All total/rollup features are in their correct "Total ..." leaves,
      NOT in __unmapped__
      - No rollup/combined features that would cause double-counting in regular
      categories
      - Features matching CRITICAL KEYWORD RULES are mapped to their specified
      categories
      - Only relevant line items are in non-__unmapped__ leaves
      - Fix until all checks pass, then output JSON only
      """

    user_content = f"""
    STRUCTURE_JSON:
    {structure_str}

    FEATURES_JSON_ARRAY:
    {features_str}

    MANDATORY MAPPINGS (APPLY BEFORE ANY OTHER RULES):
    The following exact field names MUST be mapped and NEVER placed in __unmapped__:
    1. "operating expenses" -> Operating Expenses / Total Operating Expenses
    2. "revenues" -> Revenues / Total Revenues
    3. "assets" -> Assets / Total Assets
    4. "liabilities" -> Liabilities / Total Liabilities
    5. "assets current" -> Assets / Current / Total Current Assets
    6. "liabilities current" -> Liabilities / Current / Total Current Liabilities
    7. "stockholders equity" -> Equity / Total Equity

    These are TOTALS that validate against their components. It is NORMAL and EXPECTED
    for both totals and components to exist. Failure to map these is an error.

    CRITICAL ANTI-DOUBLE-COUNTING RULE:
    - Before mapping any feature, scan the full feature list for potential rollups
    - If a feature name combines multiple concepts with "and"
    (e.g., "cash cash equivalents and short term investments"),
    check if those concepts exist as separate features
    - If components exist separately, the combined feature is a ROLLUP -
    map it to __unmapped__ to prevent double-counting
    - EXCEPTION: "operating expenses" and "revenues" are NOT rollups even if
    their components exist - these are validation totals

    ROLLUP EXAMPLES:
    - "costs and expenses" -> __unmapped__ (combines COGS + OpEx)
    - "cash and investments" -> __unmapped__ (if both exist separately)

    NOT ROLLUPS (must be mapped):
    - "operating expenses" (total of SG&A + R&D + other)
    - "revenues" (alternate taxonomy or validation total)
    - Any field matching MANDATORY MAPPINGS above

    APPLY KEYWORD MATCHING RULES:
    - Features containing "before income taxes" or "before taxes"
    -> Income Before Taxes (not __unmapped__)
    - Features exactly matching "nonoperating income expense"
    -> Total Non-Operating Items (not __unmapped__)
    - Features matching "costs and expenses" -> __unmapped__ (this IS a rollup)
    - Features exactly matching "operating expenses" -> Total Operating Expenses
    - Features exactly matching "revenues" -> Total Revenues

    DECISION PROCESS:
    1. Check MANDATORY MAPPINGS first - if field matches, map immediately
    2. Check if field is a true rollup (combines concepts with "and")
    3. Apply KEYWORD MATCHING RULES
    4. Be conservative for remaining fields

    Return ONLY the updated JSON with features mapped to appropriate categories.
    Map all MANDATORY fields without exception.
    """

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages
