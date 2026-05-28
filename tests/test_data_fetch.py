import unittest

from initial_trd.data_fetch import _dedupe_tickers, _parse_tickers


class DataFetchTests(unittest.TestCase):
    def test_parse_tickers_strips_empty_values(self):
        self.assertEqual(_parse_tickers("THYAO.IS, PGSUS.IS, "), ("THYAO.IS", "PGSUS.IS"))

    def test_dedupe_tickers_preserves_order(self):
        self.assertEqual(
            _dedupe_tickers(("THYAO.IS", "PGSUS.IS", "THYAO.IS")),
            ["THYAO.IS", "PGSUS.IS"],
        )


if __name__ == "__main__":
    unittest.main()
