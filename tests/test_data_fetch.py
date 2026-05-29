import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from initial_trd.data_fetch import (
    _dedupe_tickers,
    _load_cds_csv,
    _parse_observation_csv,
    _parse_tcmb_policy_rate_html,
    _parse_tcmb_cpi_html,
    _parse_tickers,
    fetch_and_align_data,
)


class DataFetchTests(unittest.TestCase):
    def test_parse_tickers_strips_empty_values(self):
        self.assertEqual(_parse_tickers("THYAO.IS, PGSUS.IS, "), ("THYAO.IS", "PGSUS.IS"))

    def test_dedupe_tickers_preserves_order(self):
        self.assertEqual(
            _dedupe_tickers(("THYAO.IS", "PGSUS.IS", "THYAO.IS")),
            ["THYAO.IS", "PGSUS.IS"],
        )

    def test_fetch_and_align_data_does_not_accept_random_state(self):
        signature = inspect.signature(fetch_and_align_data)

        self.assertNotIn("random_state", signature.parameters)

    def test_parse_tcmb_cpi_html_extracts_annual_rates(self):
        html = """
        <table>
            <tr><td>04-2026</td><td>32.37</td><td>4.18</td></tr>
            <tr><td>03-2026</td><td>30.87</td><td>1.94</td></tr>
        </table>
        """

        result = _parse_tcmb_cpi_html(html)

        self.assertEqual(
            result["date"].tolist(),
            [pd.Timestamp("2026-03-01"), pd.Timestamp("2026-04-01")],
        )
        self.assertEqual(result["CPI"].tolist(), [30.87, 32.37])

    def test_parse_observation_csv_accepts_fred_format(self):
        csv_text = (
            "observation_date,IRSTCI01TRM156N\n"
            "2026-02-01,35.500\n"
            "2026-03-01,.\n"
        )

        result = _parse_observation_csv(csv_text, output_column="CBRT_Rate")

        self.assertEqual(result["date"].tolist(), [pd.Timestamp("2026-02-01")])
        self.assertEqual(result["CBRT_Rate"].tolist(), [35.5])

    def test_parse_tcmb_policy_rate_html_extracts_one_week_repo_rates(self):
        html = """
        <table>
            <tr><td>20.05.2010</td><td>-</td><td>7.00</td></tr>
            <tr><td>23.01.2026</td><td>-</td><td>37.00</td></tr>
        </table>
        """

        result = _parse_tcmb_policy_rate_html(html)

        self.assertEqual(
            result["date"].tolist(),
            [pd.Timestamp("2010-05-20"), pd.Timestamp("2026-01-23")],
        )
        self.assertEqual(result["CBRT_Rate"].tolist(), [7.0, 37.0])

    def test_load_cds_csv_accepts_investing_export(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "turkey_5y_cds.csv"
            path.write_text(
                "Date,Price,Open,High,Low,Change %\n"
                '"May 25, 2026",250.71,250.71,250.78,250.71,-1.79%\n',
                encoding="utf-8",
            )

            result = _load_cds_csv(path)

        self.assertEqual(result["date"].tolist(), [pd.Timestamp("2026-05-25")])
        self.assertEqual(result["5Y_CDS_Spread"].tolist(), [250.71])


if __name__ == "__main__":
    unittest.main()
