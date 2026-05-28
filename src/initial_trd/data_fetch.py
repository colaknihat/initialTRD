import argparse
from datetime import timedelta

import pandas as pd
import numpy as np
import yfinance as yf

from initial_trd.paths import DATA_DIR, RAW_MARKET_PATH, STOCK_A_PATH, STOCK_B_PATH

START_DATE = "2014-01-01"
END_DATE = "2026-05-28"
DEFAULT_MARKET_TICKER = "XU100.IS"
DEFAULT_FX_TICKER = "USDTRY=X"
DEFAULT_STOCK_A_TICKER = "THYAO.IS"
DEFAULT_STOCK_B_TICKER = "PGSUS.IS"
DEFAULT_BREADTH_TICKERS = (
    "THYAO.IS",
    "ASELS.IS",
    "SISE.IS",
    "KCHOL.IS",
    "SAHOL.IS",
    "TUPRS.IS",
    "AKBNK.IS",
    "SASA.IS",
    "TOASO.IS",
    "PGSUS.IS",
)


def _download_close(tickers, *, start_date: str, end_date: str):
    data = yf.download(tickers, start=start_date, end=end_date)
    close = data["Close"]
    if isinstance(close, pd.DataFrame) and isinstance(tickers, str):
        close = close.squeeze("columns")
    return close


def fetch_and_align_data(
    *,
    stock_a_ticker: str = DEFAULT_STOCK_A_TICKER,
    stock_b_ticker: str = DEFAULT_STOCK_B_TICKER,
    breadth_tickers: tuple[str, ...] = DEFAULT_BREADTH_TICKERS,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    random_state: int | None = None,
) -> None:
    print("1. Fetching Daily Market Data (BIST100 & USD/TRY)...")
    # XU100.IS is BIST 100. USDTRY=X is the exchange rate.
    bist = _download_close(
        DEFAULT_MARKET_TICKER,
        start_date=start_date,
        end_date=end_date,
    ).rename("BIST100")
    fx = _download_close(
        DEFAULT_FX_TICKER,
        start_date=start_date,
        end_date=end_date,
    ).rename("USD_TRY")
    
    daily_df = pd.concat([bist, fx], axis=1).dropna()
    daily_df.index.name = "date"

    print("2. Calculating Market Breadth Proxy (Top 10 BIST Stocks)...")
    # Since historical advancing/declining data is expensive, we proxy it 
    # using the top 10 most liquid BIST stocks.
    tickers = _dedupe_tickers((*breadth_tickers, stock_a_ticker, stock_b_ticker))
    stocks_data = _download_close(tickers, start_date=start_date, end_date=end_date)
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
    rng = np.random.default_rng(random_state)
    
    # Dummy CPI Data (Released monthly, usually 3rd of the month)
    cpi_dates = pd.date_range(start_date, end_date, freq="MS")
    cpi_df = pd.DataFrame({"date": cpi_dates, "CPI": rng.uniform(10, 80, len(cpi_dates))})
    
    # Dummy CBRT Rate Data (Irregular meetings)
    cbrt_dates = pd.date_range(start_date, end_date, freq=pd.offsets.MonthEnd(2))
    cbrt_df = pd.DataFrame({"date": cbrt_dates, "CBRT_Rate": rng.uniform(8, 50, len(cbrt_dates))})
    
    # Dummy CDS Data (Daily, but let's assume you have a CSV)
    cds_df = daily_df[["date"]].copy()
    cds_df["5Y_CDS_Spread"] = rng.uniform(200, 600, len(cds_df)) # Proxy values

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
    stocks_data[[stock_a_ticker]].dropna().rename(columns={stock_a_ticker: "close"}).to_csv(STOCK_A_PATH)
    stocks_data[[stock_b_ticker]].dropna().rename(columns={stock_b_ticker: "close"}).to_csv(STOCK_B_PATH)
    
    print(f"Pair: stock_A={stock_a_ticker}, stock_B={stock_b_ticker}")
    print(f"Success! Data saved to {DATA_DIR}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and align BIST, USD/TRY, breadth, and macro proxy data."
    )
    parser.add_argument("--stock-a-ticker", default=DEFAULT_STOCK_A_TICKER)
    parser.add_argument("--stock-b-ticker", default=DEFAULT_STOCK_B_TICKER)
    parser.add_argument(
        "--breadth-tickers",
        default=",".join(DEFAULT_BREADTH_TICKERS),
        help="Comma-separated tickers used for the market breadth proxy.",
    )
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Seed for generated macro proxy data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fetch_and_align_data(
        stock_a_ticker=args.stock_a_ticker,
        stock_b_ticker=args.stock_b_ticker,
        breadth_tickers=_parse_tickers(args.breadth_tickers),
        start_date=args.start_date,
        end_date=args.end_date,
        random_state=args.random_state,
    )


def _parse_tickers(value: str) -> tuple[str, ...]:
    tickers = tuple(ticker.strip() for ticker in value.split(",") if ticker.strip())
    if not tickers:
        raise ValueError("at least one ticker is required")
    return tickers


def _dedupe_tickers(tickers: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(tickers))


if __name__ == "__main__":
    main()
