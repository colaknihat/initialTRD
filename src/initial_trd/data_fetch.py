import argparse
from datetime import timedelta
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
import re
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

from initial_trd.paths import (
    DATA_DIR,
    RAW_MARKET_PATH,
    STOCK_A_PATH,
    STOCK_B_PATH,
    resolve_project_path,
)

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
DEFAULT_TURKSTAT_CPI_URL = (
    "https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB%20EN/Main%20Menu/"
    "Statistics/Inflation%20Data/Consumer%20Prices"
)
DEFAULT_CBRT_RATE_SERIES = "IRSTCI01TRM156N"
DEFAULT_FRED_CBRT_RATE_URL = (
    f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={DEFAULT_CBRT_RATE_SERIES}"
)
DEFAULT_CBRT_RATE_URL = (
    "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB%20TR/Main%20Menu/"
    "Temel%20Faaliyetler/Para%20Politikasi/"
    "Merkez%20Bankasi%20Faiz%20Oranlari/1%20Hafta%20Repo"
)
DEFAULT_CDS_CSV_PATH = DATA_DIR / "turkey_5y_cds.csv"
URL_TIMEOUT_SECONDS = 30
USER_AGENT = "initial-trd/0.1"


class _HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag == "td" and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._current_cell is not None:
            cell = " ".join("".join(self._current_cell).split())
            if self._current_row is not None:
                self._current_row.append(cell)
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None


def _download_close(tickers, *, start_date: str, end_date: str):
    data = yf.download(tickers, start=start_date, end=end_date)
    close = data["Close"]
    if isinstance(close, pd.DataFrame) and isinstance(tickers, str):
        close = close.squeeze("columns")
    return close


def fetch_stock_closes(
    tickers: tuple[str, ...],
    *,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """Download one shared close-price frame for a stock universe."""

    ticker_list = _dedupe_tickers(tickers)
    if len(ticker_list) < 2:
        raise ValueError("at least two tickers are required")

    closes = _download_close(ticker_list, start_date=start_date, end_date=end_date)
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(ticker_list[0])

    closes = closes.reindex(columns=ticker_list)
    closes.index = pd.to_datetime(closes.index)
    closes.index.name = "date"
    return closes.sort_index()


def fetch_and_align_data(
    *,
    stock_a_ticker: str = DEFAULT_STOCK_A_TICKER,
    stock_b_ticker: str = DEFAULT_STOCK_B_TICKER,
    breadth_tickers: tuple[str, ...] = DEFAULT_BREADTH_TICKERS,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    cpi_url: str = DEFAULT_TURKSTAT_CPI_URL,
    cbrt_rate_url: str = DEFAULT_CBRT_RATE_URL,
    cds_csv: str | Path = DEFAULT_CDS_CSV_PATH,
) -> None:
    cds_csv_path = resolve_project_path(cds_csv)
    if not cds_csv_path.exists():
        raise FileNotFoundError(_missing_cds_csv_message(cds_csv_path))

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
    cpi_df = fetch_tcmb_cpi_data(cpi_url)
    cbrt_df = fetch_cbrt_rate_data(cbrt_rate_url)
    cds_df = _load_cds_csv(cds_csv_path)

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


def fetch_tcmb_cpi_data(cpi_url: str = DEFAULT_TURKSTAT_CPI_URL) -> pd.DataFrame:
    """Fetch annual CPI inflation published by CBRT from TURKSTAT data."""

    return _parse_tcmb_cpi_html(_read_url_text(cpi_url))


def fetch_cbrt_rate_data(cbrt_rate_url: str = DEFAULT_CBRT_RATE_URL) -> pd.DataFrame:
    """Fetch the CBRT policy-rate series from TCMB HTML or FRED-style CSV."""

    text = _read_url_text(cbrt_rate_url)
    if _looks_like_csv(text):
        return _parse_observation_csv(text, output_column="CBRT_Rate")
    return _parse_tcmb_policy_rate_html(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and align BIST, USD/TRY, breadth, and real macro data."
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
        "--cpi-url",
        default=DEFAULT_TURKSTAT_CPI_URL,
        help="TCMB/TURKSTAT CPI page URL.",
    )
    parser.add_argument(
        "--cbrt-rate-url",
        default=DEFAULT_CBRT_RATE_URL,
        help="TCMB policy-rate page URL or FRED-compatible CSV URL.",
    )
    parser.add_argument(
        "--cds-csv",
        default=str(DEFAULT_CDS_CSV_PATH),
        help=(
            "Turkey 5Y CDS CSV from Bloomberg, Refinitiv, or Investing.com. "
            "Expected columns: Date and Price/Close/PX_LAST/5Y_CDS_Spread."
        ),
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
        cpi_url=args.cpi_url,
        cbrt_rate_url=args.cbrt_rate_url,
        cds_csv=args.cds_csv,
    )


def _read_url_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=URL_TIMEOUT_SECONDS) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _parse_tcmb_cpi_html(html_text: str) -> pd.DataFrame:
    parser = _HtmlTableParser()
    parser.feed(html_text)

    records = []
    for row in parser.rows:
        if len(row) < 2 or not re.fullmatch(r"\d{2}-\d{4}", row[0]):
            continue
        records.append(
            {
                "date": pd.to_datetime(row[0], format="%m-%Y"),
                "CPI": _parse_number(row[1]),
            }
        )

    if not records:
        raise ValueError("no TURKSTAT CPI rows were found in the TCMB CPI page")

    return _clean_time_series(pd.DataFrame(records), "CPI")


def _parse_tcmb_policy_rate_html(html_text: str) -> pd.DataFrame:
    parser = _HtmlTableParser()
    parser.feed(html_text)

    records = []
    for row in parser.rows:
        if len(row) < 2 or not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", row[0]):
            continue
        records.append(
            {
                "date": pd.to_datetime(row[0], format="%d.%m.%Y"),
                "CBRT_Rate": _parse_number(row[-1]),
            }
        )

    if not records:
        raise ValueError("no one-week repo policy-rate rows were found in the TCMB page")

    return _clean_time_series(pd.DataFrame(records), "CBRT_Rate")


def _parse_observation_csv(csv_text: str, *, output_column: str) -> pd.DataFrame:
    source = pd.read_csv(StringIO(csv_text), na_values=[".", ""])
    date_column = _find_column(source, ("date", "observation_date", "DATE"))
    value_columns = [column for column in source.columns if column != date_column]
    if not value_columns:
        raise ValueError("rate CSV must include one value column")

    value_column = (
        output_column if output_column in source.columns else value_columns[0]
    )
    result = source[[date_column, value_column]].rename(
        columns={date_column: "date", value_column: output_column}
    )
    return _clean_time_series(result, output_column)


def _looks_like_csv(text: str) -> bool:
    first_line = text.lstrip("\ufeff\r\n\t ").splitlines()[0]
    return "," in first_line and "<html" not in first_line.lower()


def _load_cds_csv(cds_csv: str | Path) -> pd.DataFrame:
    path = resolve_project_path(cds_csv)
    if not path.exists():
        raise FileNotFoundError(_missing_cds_csv_message(path))

    source = pd.read_csv(path)
    date_column = _find_column(source, ("date", "observation_date", "DATE"))
    value_column = _find_column(
        source,
        (
            "5Y_CDS_Spread",
            "cds",
            "CDS",
            "Turkey_CDS_5Y",
            "Price",
            "Close",
            "Last",
            "Last Price",
            "PX_LAST",
            "TRGV5YUSAC=R",
        ),
    )
    result = source[[date_column, value_column]].rename(
        columns={date_column: "date", value_column: "5Y_CDS_Spread"}
    )
    return _clean_time_series(result, "5Y_CDS_Spread")


def _clean_time_series(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned[value_column] = cleaned[value_column].map(_parse_number)
    cleaned = cleaned.dropna(subset=["date", value_column])
    cleaned = cleaned.drop_duplicates(subset=["date"], keep="last")
    return cleaned.sort_values("date").reset_index(drop=True)


def _parse_number(value) -> float:
    if pd.isna(value):
        return float("nan")

    text = str(value).strip().replace("%", "").replace("\u2212", "-")
    text = re.sub(r"[^\d,.\-+]", "", text)
    if not text:
        return float("nan")
    if "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    return float(text)


def _find_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str:
    columns_by_lower = {str(column).strip().lower(): column for column in df.columns}
    for alias in aliases:
        column = columns_by_lower.get(alias.lower())
        if column is not None:
            return column
    raise ValueError(
        "CSV is missing one of the expected columns: " + ", ".join(aliases)
    )


def _missing_cds_csv_message(path: Path) -> str:
    return (
        f"{path} does not exist. Download Turkey 5Y CDS historical data from "
        "Bloomberg, Refinitiv, or Investing.com and pass --cds-csv. Expected "
        "columns: Date plus one of Price, Close, PX_LAST, or 5Y_CDS_Spread."
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
