#!/usr/bin/env python3
"""Render a LaTeX appendix table from profiling CSV data."""

import argparse
from pathlib import Path

import pandas as pd


def fmt_int(x):
    return "" if pd.isna(x) else f"{int(x):,}"


def fmt_float(x, nd=3):
    return "" if pd.isna(x) else f"{x:.{nd}f}"


def fmt_sps(x):
    return "" if pd.isna(x) else f"{x:,.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods_csv", default="figs/methods_table.csv")
    ap.add_argument("--out_tex", default="docs/Appendix_Methods_Table.tex")
    ap.add_argument("--title", default="Appendix A — Numerical Performance and Resources")
    ap.add_argument("--label", default="tab:methods")
    ap.add_argument("--sort", action="store_true")
    ap.add_argument("--median", action="store_true", help="Append a median summary row")
    args = ap.parse_args()

    df = pd.read_csv(args.methods_csv)

    if "steps_per_sec" not in df.columns and {"steps", "seconds"} <= set(df.columns):
        df["steps_per_sec"] = df["steps"] / df["seconds"]

    cols = [
        "N",
        "L",
        "dt",
        "steps",
        "seconds",
        "steps_per_sec",
        "max_rss_GB",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    cols = [c for c in cols if c in df.columns]

    if args.sort:
        by = [c for c in ["N", "L", "dt"] if c in df.columns]
        if by:
            df = df.sort_values(by=by)

    df = df[cols]

    # Optional median row (safe for all-NaN columns)
    med_row = None
    if args.median:
        med_vals = {}
        for c in cols:
            if c == "file":
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                s = df[c].dropna()
                if len(s):
                    med_vals[c] = float(s.median())
        if med_vals:
            row = []
            if "N" in cols:
                val = med_vals.get("N")
                row.append(f"{int(val)}" if val is not None else "")
            if "L" in cols:
                val = med_vals.get("L")
                row.append(f"{val:.0f}" if val is not None else "")
            if "dt" in cols:
                val = med_vals.get("dt")
                row.append(f"{val:.7f}" if val is not None else "")
            if "steps" in cols:
                val = med_vals.get("steps")
                row.append(f"{int(val)}" if val is not None else "")
            if "seconds" in cols:
                val = med_vals.get("seconds")
                row.append(f"{val:.1f}" if val is not None else "")
            if "steps_per_sec" in cols:
                val = med_vals.get("steps_per_sec")
                row.append(fmt_sps(val).replace(",", "\\,") if val is not None else "")
            if "max_rss_GB" in cols:
                val = med_vals.get("max_rss_GB")
                row.append(f"{val:.2f}" if val is not None else "")
            if "OMP_NUM_THREADS" in cols:
                val = med_vals.get("OMP_NUM_THREADS")
                row.append(f"{int(val)}" if val is not None else "")
            if "MKL_NUM_THREADS" in cols:
                val = med_vals.get("MKL_NUM_THREADS")
                row.append(f"{int(val)}" if val is not None else "")
            if "file" in cols:
                row.append("\\emph{median}")
            med_row = row

    headers = {
        "N": "N",
        "L": "L",
        "dt": "$\\Delta t$",
        "steps": "Steps",
        "seconds": "Time (s)",
        "steps_per_sec": "Steps/s",
        "max_rss_GB": "Max RSS (GB)",
        "OMP_NUM_THREADS": "OMP",
        "MKL_NUM_THREADS": "MKL",
    }

    fmt_funcs = {
        "N": fmt_int,
        "L": lambda x: fmt_float(x, nd=0),
        "dt": lambda x: f"{x:.7f}" if not pd.isna(x) else "",
        "steps": fmt_int,
        "seconds": lambda x: fmt_float(x, nd=1),
        "steps_per_sec": fmt_sps,
        "max_rss_GB": lambda x: fmt_float(x, nd=2),
        "OMP_NUM_THREADS": fmt_int,
        "MKL_NUM_THREADS": fmt_int,
    }

    aligns = {
        "N": "r",
        "L": "r",
        "dt": "r",
        "steps": "r",
        "seconds": "r",
        "steps_per_sec": "r",
        "max_rss_GB": "r",
        "OMP_NUM_THREADS": "c",
        "MKL_NUM_THREADS": "c",
    }

    columns = df.columns.tolist()
    col_spec = "".join(aligns.get(c, "c") for c in columns)

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{args.title}}}")
    lines.append(f"\\label{{{args.label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    lines.append(" & ".join(headers[c] for c in columns) + "\\\\")
    lines.append("\\hline")

    def fmt_row(row):
        return " & ".join(fmt_funcs.get(c, str)(row[c]) for c in columns) + "\\\\"

    for _, r in df.iterrows():
        lines.append(fmt_row(r))

    if med_row is not None:
        lines.append("\\hline")
        lines.append(" & ".join(med_row) + "\\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_path = Path(args.out_tex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[appendix-tex] wrote {out_path} ({len(df)} rows{' + median' if med_row is not None else ''})")


if __name__ == "__main__":
    main()
