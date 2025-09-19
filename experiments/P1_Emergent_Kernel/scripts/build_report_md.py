#!/usr/bin/env python3
# Build a single REPORT.md that stitches acceptance, diagnostics, and methods,
# now with YAML front-matter (title, UTC timestamp, git info).
import argparse, json, glob, os, subprocess
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

def read_text(path):
    p = Path(path)
    return p.read_text() if p.exists() else None

def maybe_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None

def section(title): return f"\n## {title}\n"

def df_to_md(df, max_rows=20):
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    try:
        return d.to_markdown(index=False)
    except Exception:
        header = "| " + " | ".join(d.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(d.columns)) + " |"
        rows = []
        for _, row in d.iterrows():
            cells = [str(row[c]) if pd.notna(row[c]) else "" for c in d.columns]
            rows.append("| " + " | ".join(cells) + " |")
        body = "\n".join(rows) if rows else "| _no data_ |"
        return "\n".join([header, sep, body])

def _git(cmd):
    try:
        out = subprocess.run(["git"] + cmd, check=False, capture_output=True, text=True)
        return out.stdout.strip() or None
    except Exception:
        return None

def git_info():
    # Prefer environment (CI) if available
    env_commit = os.environ.get("CI_COMMIT_SHA") or os.environ.get("GIT_COMMIT")
    env_branch = os.environ.get("CI_COMMIT_BRANCH") or os.environ.get("GIT_BRANCH")
    env_tag    = os.environ.get("CI_COMMIT_TAG") or os.environ.get("GIT_TAG")
    env_repo   = os.environ.get("CI_REPOSITORY_URL") or os.environ.get("GIT_URL")
    info = {
        "commit": env_commit or _git(["rev-parse", "HEAD"]),
        "short":  _git(["rev-parse", "--short", "HEAD"]),
            # if env_commit exists but 'short' is None, derive a short fallback
    }
    if info["short"] is None and info["commit"]:
        info["short"] = info["commit"][:7]
    info["branch"] = env_branch or _git(["rev-parse", "--abbrev-ref", "HEAD"])
    info["tag"]    = env_tag    or _git(["describe", "--tags", "--abbrev=0"])
    info["remote"] = env_repo   or _git(["config", "--get", "remote.origin.url"])
    return info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="DON Emergent Gravity — Validation Report (P1)")
    ap.add_argument("--accept", default="ACCEPTANCE_BOX.md")
    ap.add_argument("--helm_json", default="figs/helmholtz_checks.json")
    ap.add_argument("--kepler_csv", default="figs/kepler_fit.csv")
    ap.add_argument("--ep_glob", default="sweeps/ep_m*/summary.csv")
    ap.add_argument("--methods_csv", default="figs/methods_table.csv")
    ap.add_argument("--out", default="docs/REPORT.md")
    args = ap.parse_args()

    # --- Front-matter (YAML)
    gi = git_info()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fm = [
        "---",
        f"title: {args.title}",
        f"date: {ts}",
        f"commit: {gi.get('commit') or 'unknown'}",
        f"commit_short: {gi.get('short') or 'unknown'}",
        f"branch: {gi.get('branch') or 'unknown'}",
        f"tag: {gi.get('tag') or '—'}",
        f"repo: {gi.get('remote') or '—'}",
        "---",
        ""
    ]

    lines = []
    lines.extend(fm)
    lines.append(f"# {args.title}\n")

    # Build metadata (reader-visible)
    lines.append(section("Build Metadata"))
    lines.append(f"- **Timestamp (UTC):** {ts}")
    lines.append(f"- **Commit:** `{gi.get('short') or 'unknown'}` ({gi.get('commit') or 'unknown'})")
    lines.append(f"- **Branch:** {gi.get('branch') or 'unknown'}")
    lines.append(f"- **Tag:** {gi.get('tag') or '—'}")
    lines.append(f"- **Repository:** {gi.get('remote') or '—'}")

    # Acceptance box
    accept = read_text(args.accept)
    lines.append(section("Acceptance Box"))
    if accept:
        lines.append(accept.strip())
    else:
        lines.append("_ACCEPTANCE_BOX.md not found_")

    # Helmholtz annulus summary
    js = Path(args.helm_json)
    lines.append(section("Helmholtz Diagnostics"))
    if js.exists():
        try:
            o = json.loads(js.read_text())
            lines.append(f"- **Window:** [{o['window'][0]}, {o['window'][1]}]")
            lines.append(f"- **R_curl (mean):** {o['Rcurl_mean']:.3e}")
            lines.append(f"- **tau_frac (mean):** {o['tau_frac_mean']:.3e}")
            lines.append(f"- **radiality error (mean):** {o['radiality_err_mean']:.3e}")
        except Exception as e:
            lines.append(f"_Failed to parse {args.helm_json}: {e}_")
    else:
        lines.append("_No helmholtz_checks.json found_")

    # Kepler
    kdf = maybe_csv(args.kepler_csv)
    lines.append(section("Kepler T²–a³ & Flatness"))
    if kdf is not None and len(kdf):
        import numpy as np
        slope = np.polyfit(np.log(kdf["a"]), np.log(kdf["T"]**2), 1)[0]
        lines.append(f"- **slope dlog(T²)/dlog(a):** **{slope:.3f}** (target 1.500±0.03)")
        lines.append(f"- **mean flatness (MAD/med):** **{kdf['flatness_mad_over_med'].mean():.3f}** (target ≤ 0.03)")
        lines.append("\n<details><summary>Kepler samples</summary>\n\n" +
                     df_to_md(kdf[["log","T","a","flatness_mad_over_med"]]) +
                     "\n\n</details>")
    else:
        lines.append("_No kepler_fit.csv found_")

    # Equivalence Principle
    lines.append(section("Equivalence Principle"))
    ep_paths = sorted(glob.glob(args.ep_glob))
    if ep_paths:
        vals = []
        for p in ep_paths:
            df = pd.read_csv(p)
            col = None
            for c in ["a_rad_mean","accel_r_mean","acc_mean","a_r_mean"]:
                if c in df.columns: col = c; break
            if col is None: continue
            m = None
            for mc in ["m_test","m4","mass_test"]:
                if mc in df.columns:
                    m = float(df[mc].iloc[0]); break
            vals.append((p, m, float(df[col].iloc[0])))
        if vals:
            import numpy as np
            a = np.array([v[2] for v in vals], float)
            rel = (a.max() - a.min()) / a.mean() if a.size else float("nan")
            lines.append(f"- **normalized accel spread Δa/a:** **{rel*100:.3f}%** (target < 0.5%)")
            epdf = pd.DataFrame(vals, columns=["file","m","a_metric"])
            lines.append("\n<details><summary>EP samples</summary>\n\n" + df_to_md(epdf) + "\n\n</details>")
        else:
            lines.append("_EP summaries found, but no accel column detected_")
    else:
        lines.append("_No EP summaries found_")

    # Methods table
    mdf = maybe_csv(args.methods_csv)
    lines.append(section("Methods — Performance & Resources"))
    if mdf is not None and len(mdf):
        lines.append(df_to_md(mdf))
    else:
        lines.append("_No methods_table.csv found_")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print(f"[report] wrote {args.out}")

if __name__ == "__main__":
    main()
