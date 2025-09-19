#!/usr/bin/env python3
"""Assemble a single Markdown report covering validation diagnostics."""

import argparse
import glob
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def read_text(path: Path) -> str:
    return path.read_text() if path.exists() else "_Not available._\n"


def markdown_table(headers: List[str], rows: Iterable[Iterable[str]]) -> str:
    hdr = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]
    return "\n".join([hdr, sep] + body_lines) if body_lines else hdr + "\n" + sep + "\n| _no data_ |"


def section_acceptance(path: Path) -> str:
    content = read_text(path).strip()
    if not content:
        return "## Acceptance Box\n\n_Not available._\n"
    return content


def section_helmholtz(csv_path: Path) -> str:
    if not csv_path.exists():
        return "## Helmholtz Diagnostics\n\n_Not available._\n"
    df = pd.read_csv(csv_path)
    cols = [
        "r_mid",
        "Rcurl_median",
        "tau_frac_median",
        "radiality_err_median",
    ]
    subset = df[cols].copy()
    stats = {
        "r_min": subset["r_mid"].min(),
        "r_max": subset["r_mid"].max(),
        "curl_med": subset["Rcurl_median"].median(),
        "tau_med": subset["tau_frac_median"].median(),
        "radiality_med": subset["radiality_err_median"].median(),
    }
    rows = []
    for _, row in subset.iterrows():
        rows.append(
            [
                f"{row.r_mid:.2f}",
                f"{row.Rcurl_median:.2e}",
                f"{row.tau_frac_median:.3f}",
                f"{row.radiality_err_median:.3f}",
            ]
        )
    return (
        "## Helmholtz Diagnostics\n\n"
        f"Radial window {stats['r_min']:.1f}–{stats['r_max']:.1f}. "
        f"Median curl {stats['curl_med']:.2e}, median |tau_frac| {stats['tau_med']:.3f}, "
        f"median radiality error {stats['radiality_med']:.3f}.\n\n"
        + markdown_table([
            "r_mid",
            "Rcurl_med",
            "tau_frac_med",
            "radiality_err_med",
        ], rows)
        + "\n"
    )


def section_kepler(csv_path: Path) -> str:
    if not csv_path.exists():
        return "## Kepler Fit\n\n_Not available._\n"
    df = pd.read_csv(csv_path)
    if df.empty:
        return "## Kepler Fit\n\n_No Kepler logs processed._\n"
    x = np.log(df["a"].to_numpy())
    y = np.log(np.square(df["T"].to_numpy()))
    slope = np.polyfit(x, y, 1)[0]
    flat_mean = df["flatness_mad_over_med"].mean()
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                Path(row["log"]).name,
                f"{row['T']:.3f}",
                f"{row['a']:.3f}",
                f"{row['flatness_mad_over_med']:.3f}",
            ]
        )
    return (
        "## Kepler Fit\n\n"
        f"Slope dlog(T^2)/dlog(a) = {slope:.3f} (target 1.500±0.03). "
        f"Mean flatness MAD/median = {flat_mean:.3f}.\n\n"
        + markdown_table([
            "log",
            "T",
            "a",
            "flatness",
        ], rows)
        + "\n"
    )


def section_ep(ep_glob: str) -> str:
    paths = sorted(glob.glob(ep_glob))
    records = []
    for path in paths:
        df = pd.read_csv(path)
        accel_cols = df.filter(regex=r"(a_rad_mean|accel_r_mean|acc_mean|slope_Fr)")
        if accel_cols.empty:
            continue
        accel = float(accel_cols.iloc[0, 0])
        mass = None
        mass_cols = df.filter(regex=r"(m_test|mass_test|m4|masses)")
        if not mass_cols.empty:
            mass = mass_cols.iloc[0, 0]
        if mass is None:
            import re

            match = re.search(r"ep_m([\d.]+)", path)
            if match:
                mass = float(match.group(1))
        if mass is None:
            continue
        records.append((float(mass), accel))
    if not records:
        return "## Equivalence Principle\n\n_No EP sweep summaries with acceleration columns found._\n"
    records.sort()
    masses, accels = zip(*records)
    accels = np.array(accels, float)
    mean_acc = accels.mean() if accels.size else float("nan")
    spread = (accels.max() - accels.min()) / mean_acc if mean_acc else float("nan")
    rows = [[f"{m:.2f}", f"{a:.6f}"] for m, a in records]
    summary = (
        "## Equivalence Principle\n\n"
        f"Accelerations across masses: normalized spread {abs(spread)*100:.3f}% (target < 0.5%).\n\n"
    )
    return summary + markdown_table(["mass", "accel"], rows) + "\n"


def section_methods_table(csv_path: Path) -> str:
    if not csv_path.exists():
        return "## Methods Table\n\n_Not available._\n"
    df = pd.read_csv(csv_path)
    if df.empty:
        return "## Methods Table\n\n_No profile runs recorded._\n"
    headers = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        rows.append(["" if pd.isna(row[h]) else f"{row[h]}" for h in headers])
    return "## Methods Table\n\n" + markdown_table(headers, rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acceptance", default="ACCEPTANCE_BOX.md")
    ap.add_argument("--helmholtz", default="figs/helmholtz_checks.csv")
    ap.add_argument("--kepler", default="figs/kepler_fit.csv")
    ap.add_argument("--ep_glob", default="sweeps/ep_m*/summary.csv")
    ap.add_argument("--methods", default="figs/methods_table.csv")
    ap.add_argument("--out", default="REPORT.md")
    args = ap.parse_args()

    parts = ["# DON Emergent Gravity Validation Report\n"]
    parts.append(section_acceptance(Path(args.acceptance)))
    parts.append(section_helmholtz(Path(args.helmholtz)))
    parts.append(section_kepler(Path(args.kepler)))
    parts.append(section_ep(args.ep_glob))
    parts.append(section_methods_table(Path(args.methods)))

    Path(args.out).write_text("\n\n".join(part.strip() for part in parts if part) + "\n")
    print(f"[report] wrote {args.out}")


if __name__ == "__main__":
    main()
