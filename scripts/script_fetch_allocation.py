import csv

def create_portfolio_csv(filename, allocations, companies):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Allocation", "Name"])
        for company, allocation in zip(companies, allocations):
            writer.writerow([allocation, company])

def main():
    companies = ["META", "AAPL", "AMZN", "NFLX", "GOOG"]
    allocations = [20.0, 30.0, 25.0, 15.0, 10.0]  # Example allocations
    create_portfolio_csv("MAANG_portfolio.csv", allocations, companies)

if __name__ == "__main__":
    main()
