#!/usr/bin/env python3
# Generates a paper-ready Markdown appendix from figs/methods_table.csv
import argparse, math
from pathlib import Path
import pandas as pd

def fmt_int(x):
    return "" if pd.isna(x) else f"{int(x):,}"

def fmt_float(x, nd=3):
    return "" if pd.isna(x) else f"{x:.{nd}f}"

def fmt_sps(x):
    # steps/sec: show with thousands separators, 1 decimal
    return "" if pd.isna(x) else f"{x:,.1f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods_csv", default="figs/methods_table.csv")
    ap.add_argument("--out_md", default="docs/Appendix_Methods_Table.md")
    ap.add_argument("--title", default="Appendix A — Numerical Performance and Resources")
    ap.add_argument("--sort", action="store_true", help="Sort rows by N, then L, then dt")
    ap.add_argument("--add_summary", action="store_true", help="Append median row")
    args = ap.parse_args()

    df = pd.read_csv(args.methods_csv)

    # Backfill steps_per_sec if missing
    if "steps_per_sec" not in df.columns and {"steps","seconds"} <= set(df.columns):
        df["steps_per_sec"] = df["steps"] / df["seconds"]

    # Order + sort
    cols = ["N","L","dt","steps","seconds","steps_per_sec","max_rss_GB","OMP_NUM_THREADS","MKL_NUM_THREADS","file"]
    cols = [c for c in cols if c in df.columns]
    if args.sort:
        by = [c for c in ["N","L","dt"] if c in df.columns]
        if by: df = df.sort_values(by=by)
    df = df[cols]

    # Optional median summary (safe for all-NaN columns)
    med = None
    if args.add_summary:
        med = {}
        for c in df.columns:
            if c in ["file"]:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                s = df[c].dropna()
                if len(s):
                    med[c] = float(s.median())
        if med:
            med["file"] = "— median —"
        else:
            med = None

    # Build Markdown
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# {args.title}\n")
    lines.append("This table summarizes numerical performance and resource usage for the strict-orbit validation runs.\n")
    lines.append("> Notes: measurements taken with `/usr/bin/time -l` on macOS; threads pinned via `OMP_NUM_THREADS`/`MKL_NUM_THREADS`. Steps/s computed as `steps / real_seconds`. Memory is the maximum resident set size (approx.).\n")

    # Header with units
    hdr = []
    if "N" in df.columns:               hdr.append("N (cells)")
    if "L" in df.columns:               hdr.append("L")
    if "dt" in df.columns:              hdr.append("Δt")
    if "steps" in df.columns:           hdr.append("Steps")
    if "seconds" in df.columns:         hdr.append("Time (s)")
    if "steps_per_sec" in df.columns:   hdr.append("Steps/s")
    if "max_rss_GB" in df.columns:      hdr.append("Max RSS (GB)")
    if "OMP_NUM_THREADS" in df.columns: hdr.append("OMP")
    if "MKL_NUM_THREADS" in df.columns: hdr.append("MKL")
    if "file" in df.columns:            hdr.append("Run")

    sep = ["---"]*len(hdr)
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    def row_fmt(r):
        rec = pd.Series(r) if isinstance(r, dict) else r

        def val(key):
            if isinstance(rec, pd.Series):
                return rec.get(key)
            return getattr(rec, key, None)

        def _fmt(x):
            if x is None:
                return ""
            if isinstance(x, float) and pd.isna(x):
                return ""
            return str(x)

        out = []
        if "N" in df.columns:
            n = val("N")
            out.append(_fmt(int(n) if n is not None and not pd.isna(n) else None))
        if "L" in df.columns:
            l = val("L")
            out.append(_fmt(round(l) if l is not None and not pd.isna(l) else None))
        if "dt" in df.columns:
            dtv = val("dt")
            out.append(_fmt(f"{dtv:.7f}" if dtv is not None and not pd.isna(dtv) else None))
        if "steps" in df.columns:
            steps = val("steps")
            out.append(_fmt(int(steps) if steps is not None and not pd.isna(steps) else None))
        if "seconds" in df.columns:
            seconds = val("seconds")
            out.append(_fmt(f"{seconds:.1f}" if seconds is not None and not pd.isna(seconds) else None))
        if "steps_per_sec" in df.columns:
            sps_val = val("steps_per_sec")
            out.append(_fmt(f"{sps_val:,.1f}" if sps_val is not None and not pd.isna(sps_val) else None))
        if "max_rss_GB" in df.columns:
            rss = val("max_rss_GB")
            out.append(_fmt(f"{rss:.2f}" if rss is not None and not pd.isna(rss) else None))
        if "OMP_NUM_THREADS" in df.columns:
            omp = val("OMP_NUM_THREADS")
            out.append(_fmt(int(omp) if omp is not None and not pd.isna(omp) else None))
        if "MKL_NUM_THREADS" in df.columns:
            mkl = val("MKL_NUM_THREADS")
            out.append(_fmt(int(mkl) if mkl is not None and not pd.isna(mkl) else None))
        if "file" in df.columns:
            out.append(_fmt(val("file")))
        return "| " + " | ".join(out) + " |"

    for r in df.to_dict("records"):
        lines.append(row_fmt(r))

    if med:
        # Add a visual separator then median row
        lines.append("| " + " | ".join(["**—**"]*len(hdr)) + " |")
        # Build a one-row DF to reuse formatter
        import pandas as _pd
        mdf = _pd.DataFrame([med]).reindex(columns=df.columns)
        lines.append(row_fmt(mdf.iloc[0]))

    # Footnotes
    lines.append("\n**Footnotes**  \n"
                 "1. Timings taken with BSD `/usr/bin/time -l` (real/user/sys, max RSS).  \n"
                 "2. Threads pinned: `OMP_NUM_THREADS`, `MKL_NUM_THREADS` (see columns).  \n"
                 "3. Steps/s = steps ÷ real seconds; each step is one integrator step.  \n"
                 "4. Memory is approximate; ru_maxrss semantics may differ across OS versions.\n")

    Path(args.out_md).write_text("\n".join(lines))
    print(f"[appendix-md] wrote {args.out_md} ({len(df)} rows{' + median' if med else ''})")

if __name__ == "__main__":
    main()
