if __name__ == "__main__":
    from src.jpm.question_1.models.valez_pareja import IncomeStatement

    from src.jpm.question_1.data.yf import FinanceIngestor

    # Example usage
    data = FinanceIngestor()
    income = data.constructor_data()
    # balance_sheet = data.balance_sheet()

    # # print(income)

    # row = income.iloc[-1]
    # income_date = income.index[0]
    # income_sample = row.to_dict()
    # balance_sample = balance_sheet.iloc[-1].to_dict()

    # prev_bs = BalanceSheet.from_dict(balance_sample)

    income_statement = IncomeStatement.from_dicts(income)
    print(income_statement)
    # print(income_statement)
    # print(income_statement.to_dict())
