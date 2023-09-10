import yfinance
import pandas 

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yfinance.download(ticker, start=start_date, end=end_date)
    return stock_data['Close']

def main():
    start_date = "2018-01-01"
    end_date = "2023-01-01"
    companies = ["META", "AAPL", "AMZN", "NFLX", "GOOG"]
    dataframes = []

    for company in companies:
        data = fetch_stock_data(company, start_date, end_date)
        dataframes.append(data)

    merged_data = pandas.concat(dataframes, axis=1, keys=companies)
    merged_data.to_csv("MAANG_stock_data.csv")

if __name__ == "__main__":
    main()
