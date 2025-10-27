if __name__ == "__main__":
    from src.jpm.question_1.data.yf import FinanceIngestor

    # Example usage
    data = FinanceIngestor()
    bs = data.income()
    # print(bs)
    # data = {
    #     "revenue": 100000.0,
    #     "cogs": 40000.0,
    #     "operating_expenses": 20000.0,
    #     "depreciation": 5000.0,
    #     "interest_expense": 3000.0,
    #     "tax_rate": 0.25,
    # }

    # income_statement = IncomeStatement.from_dict(data)
    # print(income_statement)
    # print(income_statement.to_dict())
