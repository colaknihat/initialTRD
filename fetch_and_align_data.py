import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta

from project_paths import DATA_DIR, RAW_MARKET_PATH, STOCK_A_PATH, STOCK_B_PATH

START_DATE = "2014-01-01"
END_DATE = "2026-05-28"


def _download_close(tickers):
    data = yf.download(tickers, start=START_DATE, end=END_DATE)
    close = data["Close"]
    if isinstance(close, pd.DataFrame) and isinstance(tickers, str):
        close = close.squeeze("columns")
    return close


def fetch_and_align_data():
    print("1. Fetching Daily Market Data (BIST100 & USD/TRY)...")
    # XU100.IS is BIST 100. USDTRY=X is the exchange rate.
    bist = _download_close("XU100.IS").rename("BIST100")
    fx = _download_close("USDTRY=X").rename("USD_TRY")
    
    daily_df = pd.concat([bist, fx], axis=1).dropna()
    daily_df.index.name = "date"

    print("2. Calculating Market Breadth Proxy (Top 10 BIST Stocks)...")
    # Since historical advancing/declining data is expensive, we proxy it 
    # using the top 10 most liquid BIST stocks.
    tickers = ["THYAO.IS", "ASELS.IS", "SISE.IS", "KCHOL.IS", "SAHOL.IS", 
               "TUPRS.IS", "AKBNK.IS", "SASA.IS", "TOASO.IS", "PGSUS.IS"]
    stocks_data = _download_close(tickers)
    daily_returns = stocks_data.pct_change(fill_method=None)
    
    # Count advancing (return > 0) and declining (return < 0)
    breadth_df = pd.DataFrame(
        {
            "advancing_stocks": (daily_returns > 0).sum(axis=1),
            "declining_stocks": (daily_returns < 0).sum(axis=1),
        }
    )
    # Add 1 to prevent division by zero in your feature engineering script
    breadth_df["declining_stocks"] = breadth_df["declining_stocks"].replace(0, 1)
    daily_df = daily_df.join(breadth_df, how="inner")

    # Ensure date is datetime and sorted
    daily_df = daily_df.reset_index(names="date")
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values("date")

    print("3. Loading & Aligning Macro Data (CPI, Rates, CDS)...")
    # NOTE: You must download these from FRED or Investing.com as CSVs.
    # For this script, we will create dummy monthly data structures to show the merge_asof logic.
    
    # Dummy CPI Data (Released monthly, usually 3rd of the month)
    cpi_dates = pd.date_range("2014-01-03", "2026-05-03", freq="MS")
    cpi_df = pd.DataFrame({"date": cpi_dates, "CPI": np.random.uniform(10, 80, len(cpi_dates))})
    
    # Dummy CBRT Rate Data (Irregular meetings)
    cbrt_dates = pd.date_range("2014-01-15", "2026-05-15", freq=pd.offsets.MonthEnd(2))
    cbrt_df = pd.DataFrame({"date": cbrt_dates, "CBRT_Rate": np.random.uniform(8, 50, len(cbrt_dates))})
    
    # Dummy CDS Data (Daily, but let's assume you have a CSV)
    cds_df = daily_df[["date"]].copy()
    cds_df["5Y_CDS_Spread"] = np.random.uniform(200, 600, len(cds_df)) # Proxy values

    print("4. Executing Anti-Lookahead Merges (merge_asof)...")
    # Merge CPI: The market only knows May CPI when it is published in June.
    # We shift CPI forward by 30 days to simulate publication delay.
    cpi_df["date"] = cpi_df["date"] + timedelta(days=30) 
    merged = pd.merge_asof(daily_df, cpi_df, on="date", direction="backward")
    
    # Merge CBRT Rate: Shift by 1 day to ensure we don't use the rate before the 14:00 announcement
    cbrt_df["date"] = cbrt_df["date"] + timedelta(days=1)
    merged = pd.merge_asof(merged, cbrt_df, on="date", direction="backward")
    
    # Merge CDS
    merged = pd.merge_asof(merged, cds_df, on="date", direction="backward")

    # Forward fill any remaining NaNs from holidays/weekends
    merged = merged.ffill().dropna()

    print("5. Saving Raw Data & Pair Trade Assets...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(RAW_MARKET_PATH, index=False)
    
    # Save individual stocks for your pairs trading script
    stocks_data[["THYAO.IS"]].dropna().rename(columns={"THYAO.IS": "close"}).to_csv(STOCK_A_PATH)
    stocks_data[["PGSUS.IS"]].dropna().rename(columns={"PGSUS.IS": "close"}).to_csv(STOCK_B_PATH)
    
    print(f"Success! Data saved to {DATA_DIR}.")

if __name__ == "__main__":
    fetch_and_align_data()
