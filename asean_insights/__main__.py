"""Command line: python -m asean_insights"""

import argparse

from .config import START_DATE
from .data import load_fundamentals, load_market_data
from .report import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the data and rebuild the report.")
    parser.add_argument("--start", default=START_DATE, help="first date, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="last date, YYYY-MM-DD (default: today)")
    parser.add_argument("--out", default="reports", help="output folder")
    parser.add_argument("--refresh", action="store_true", help="ignore today's cached download")
    args = parser.parse_args()

    data = load_market_data(args.start, args.end, refresh=args.refresh)
    fundamentals = load_fundamentals(refresh=args.refresh)
    summary = write_report(data, args.out, fundamentals)
    print(f"Wrote {summary} and figures in {summary.parent / 'figures'}")


if __name__ == "__main__":
    main()
