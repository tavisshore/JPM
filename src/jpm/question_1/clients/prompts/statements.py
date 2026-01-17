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
If a feature contains these keywords, map it regardless of verbose phrasing:

1. "before income taxes" OR "before taxes" OR "pretax" -> Income Before Taxes
- Example: "income loss from continuing operations before income taxes..."
  -> Income Before Taxes
2. "nonoperating income expense" (exact or close match)
   -> Total Non-Operating Items
3. "costs and expenses" OR "total costs and expenses"
   -> __unmapped__ (rollup of COGS + OpEx)
4. "fulfillment" -> Other Operating Expenses (NOT Cost of Revenue)
5. "technology and" (content/infrastructure/development)
   -> Research and Development

These patterns override general caution - apply them confidently.
"""
    elif statement_type == "balance_sheet":
        keyword_rules = """
CRITICAL KEYWORD MATCHING RULES FOR BALANCE SHEET (APPLY WITH CONFIDENCE):
If a feature contains these keywords, map it regardless of verbose phrasing:

1. Exact "assets" (not "assets current" or "assets noncurrent")
   -> Total Assets
2. Exact "liabilities" (not "liabilities current" or "liabilities noncurrent")
   -> Total Liabilities
3. "stockholders equity" OR "shareholders equity" -> Total Equity
4. "shares issued" OR "shares outstanding" OR "shares authorized"
   -> __unmapped__ (share counts, not dollars)
5. Features combining multiple concepts
   (e.g., "cash cash equivalents and short term investments")
   -> Check if components exist separately, if yes -> __unmapped__ (rollup)

These patterns override general caution - apply them confidently.
"""
    elif statement_type == "cash_flow":
        keyword_rules = """
CRITICAL KEYWORD MATCHING RULES FOR CASH FLOW (APPLY WITH CONFIDENCE):
If a feature contains these keywords, map it regardless of verbose phrasing:

1. "net cash provided by operating" OR "net cash from operating"
   -> Net Cash from Operating Activities
2. "net cash used in investing" OR "net cash from investing"
   -> Net Cash from Investing Activities
3. "net cash provided by financing" OR "net cash from financing"
   -> Net Cash from Financing Activities
4. "changes in working capital" OR "changes in operating assets and liabilities"
   -> Changes in Operating Assets and Liabilities
5. "acquisitions net of cash acquired"
   (with or without additional text like "and purchases of...")
   -> Acquisitions (net of cash acquired)
6. "cash" AND "period increase decrease" -> Net Change in Cash
7. "cash equivalents" AND "increase decrease including exchange rate"
   -> Net Change in Cash

These patterns override general caution and rollup detection -
apply them confidently.
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

CRITICAL ANTI-DOUBLE-COUNTING RULE:
- Before mapping any feature, scan the full feature list for potential rollups
- If a feature name combines multiple concepts
  (e.g., "cash cash equivalents and short term investments"),
  check if those concepts exist as separate features
- If the components exist separately, the combined feature is a ROLLUP -
  map it to __unmapped__ to prevent double-counting
- Example: "cash cash equivalents and short term investments" should go to
  __unmapped__ if you see both "cash and cash equivalents" AND
  "short term investments" as separate features

APPLY KEYWORD MATCHING RULES:
- Features containing "before income taxes" or "before taxes"
  -> Income Before Taxes (not __unmapped__)
- Features exactly matching "nonoperating income expense"
  -> Total Non-Operating Items (not __unmapped__)
- Features matching "costs and expenses" -> __unmapped__ (this is a rollup)

Return ONLY the updated JSON with features mapped to appropriate categories.
CRITICAL: Features like "assets", "liabilities", "assets current" MUST go to
their Total leaves, NOT __unmapped__.
Be conservative for non-total line items EXCEPT when they match CRITICAL
KEYWORD RULES - those should be mapped confidently.
"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages
