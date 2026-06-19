"""Command-line interface for missing_imputation.

Examples
--------
Impute a CSV with KNN, writing the completed table to a new file::

    missing-impute impute input.csv -o imputed.csv --method knn

Generate a synthetic demo dataset::

    missing-impute demo -o demo.csv --patients 60 --missing 0.2

List available methods::

    missing-impute methods
"""

import argparse
import sys

from . import METHODS, __version__
from .columns import FACTE_COLUMNS
from .synthetic import make_synthetic_facte


def _read_table(path):
    import pandas as pd

    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _cmd_impute(args):
    df = _read_table(args.input)
    if args.columns:
        columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    else:
        # Default to FACT-E item columns that are present in the table.
        columns = [c for c in FACTE_COLUMNS if c in df.columns]
        if not columns:
            print(
                "No --columns given and no FACT-E item columns found in the "
                "input. Specify columns with --columns col1,col2,...",
                file=sys.stderr,
            )
            return 2
    method = METHODS[args.method]
    print(f"Imputing {len(columns)} column(s) with method '{args.method}'...")
    imputed, _ = method(df, columns)
    imputed.to_csv(args.output, index=False)
    print(f"Wrote imputed data to {args.output}")
    return 0


def _cmd_demo(args):
    df = make_synthetic_facte(
        n_patients=args.patients, n_visits=args.visits, missing_rate=args.missing
    )
    df.to_csv(args.output, index=False)
    print(
        f"Wrote synthetic demo data ({len(df)} rows, "
        f"{args.patients} patients) to {args.output}"
    )
    return 0


def _cmd_methods(args):
    print("Available imputation methods:")
    for name in METHODS:
        print(f"  - {name}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="missing-impute",
        description="Impute missing values in longitudinal clinical PRO data.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_impute = sub.add_parser("impute", help="Impute missing values in a CSV/XLSX file.")
    p_impute.add_argument("input", help="Input CSV or XLSX file.")
    p_impute.add_argument("-o", "--output", required=True, help="Output CSV path.")
    p_impute.add_argument(
        "-m", "--method", choices=sorted(METHODS), default="knn", help="Imputation method."
    )
    p_impute.add_argument(
        "--columns",
        help="Comma-separated columns to impute. Defaults to FACT-E items present in the file.",
    )
    p_impute.set_defaults(func=_cmd_impute)

    p_demo = sub.add_parser("demo", help="Generate a synthetic demo dataset.")
    p_demo.add_argument("-o", "--output", required=True, help="Output CSV path.")
    p_demo.add_argument("--patients", type=int, default=60, help="Number of patients.")
    p_demo.add_argument("--visits", type=int, default=5, help="Visits per patient.")
    p_demo.add_argument("--missing", type=float, default=0.15, help="Missing-value rate.")
    p_demo.set_defaults(func=_cmd_demo)

    p_methods = sub.add_parser("methods", help="List available imputation methods.")
    p_methods.set_defaults(func=_cmd_methods)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
