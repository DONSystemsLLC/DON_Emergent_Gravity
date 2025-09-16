#!/usr/bin/env python3
# Cross-platform parser for /usr/bin/time -l (BSD/macOS) and gtime -v (GNU)
import argparse, glob, json, re
from pathlib import Path
import pandas as pd

def _parse_bsd_timelog(txt: str):
    """Parse BSD `/usr/bin/time -l` output."""
    # real/user/sys show up as "X.Y real", "X.Y user", "X.Y sys"
    def grab_float(pattern):
        m = re.search(pattern, txt, flags=re.IGNORECASE|re.MULTILINE)
        return float(m.group(1)) if m else None
    real = grab_float(r'^\s*([\d\.]+)\s+real')
    user = grab_float(r'^\s*([\d\.]+)\s+user')
    sys  = grab_float(r'^\s*([\d\.]+)\s+sys')

    # max RSS line: "123456  maximum resident set size"
    rss_bytes = None
    m = re.search(r'maximum resident set size\s+(\d+)', txt, flags=re.IGNORECASE)
    if m:
        val = int(m.group(1))
        # Heuristic: On macOS ru_maxrss often prints bytes; if value is small, assume KB
        rss_bytes = val if val > 10_000_000 else val * 1024
    return {"real_s": real, "user_s": user, "sys_s": sys, "max_rss_bytes": rss_bytes}

def _hms_to_seconds(s: str):
    # Accept h:mm:ss, m:ss, or seconds with optional decimals
    s = s.strip()
    if re.match(r'^\d+(\.\d+)?$', s):
        return float(s)
    parts = s.split(':')
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        h,m,sec = parts
        return h*3600 + m*60 + sec
    if len(parts) == 2:
        m,sec = parts
        return m*60 + sec
    raise ValueError(f"Unrecognized elapsed format: {s}")

def _parse_gnu_timelog(txt: str):
    """Parse GNU `gtime -v` output."""
    def grab_float(label):
        m = re.search(rf'^{re.escape(label)}:\s*([\d\.]+)\s*$', txt, flags=re.MULTILINE)
        return float(m.group(1)) if m else None

    user = grab_float("User time (seconds)")
    sys  = grab_float("System time (seconds)")

    # Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.58
    m = re.search(r'^Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)\s*$', txt, flags=re.MULTILINE)
    real = _hms_to_seconds(m.group(1)) if m else None

    # Maximum resident set size (kbytes): 123456
    m = re.search(r'^Maximum resident set size \(kbytes\):\s*(\d+)\s*$', txt, flags=re.MULTILINE)
    rss_bytes = int(m.group(1))*1024 if m else None
    if rss_bytes is None:
        m = re.search(r'^(?:Resident Set Size|Average resident set size).*?:\s*(\d+)\s*$', txt, flags=re.MULTILINE | re.IGNORECASE)
        if m:
            rss_bytes = int(m.group(1)) * 1024

    return {"real_s": real, "user_s": user, "sys_s": sys, "max_rss_bytes": rss_bytes}

def parse_timelog(path: Path):
    txt = path.read_text()
    # Decide parser by signature
    if "Elapsed (wall clock) time" in txt or "User time (seconds)" in txt:
        return _parse_gnu_timelog(txt)
    if "maximum resident set size" in txt or re.search(r'^\s*[\d\.]+\s+real', txt, re.MULTILINE):
        return _parse_bsd_timelog(txt)
    # Fallback: try to salvage elapsed/user/sys lines generically
    real = None
    m = re.search(r'^\s*([\d\.]+)\s+real', txt, flags=re.MULTILINE)
    if m: real = float(m.group(1))
    return {"real_s": real, "user_s": None, "sys_s": None, "max_rss_bytes": None}

def load_meta(path_timelog: Path):
    meta_path = path_timelog.with_suffix(".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    # Fallback from filename: N###_L###_dtX.Y
    m = re.search(r'N(\d+)_L(\d+)_dt([0-9.]+)', path_timelog.name)
    meta = {}
    if m:
        meta.update({"N": int(m.group(1)), "L": float(m.group(2)), "dt": float(m.group(3))})
    return meta

def gb(bytes_val):
    return None if bytes_val is None else bytes_val / (1024**3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="profile_runs/*.timelog")
    ap.add_argument("--out", default="figs/methods_table.csv")
    args = ap.parse_args()

    rows = []
    for tl in sorted(glob.glob(args.glob)):
        tlp = Path(tl)
        tm  = parse_timelog(tlp)
        meta = load_meta(tlp)
        steps  = meta.get("steps") or None
        real_s = tm.get("real_s")
        sps    = (steps / real_s) if (steps and real_s) else None
        rows.append({
            "file": tlp.name,
            "N": meta.get("N"), "L": meta.get("L"), "dt": meta.get("dt"),
            "steps": steps, "seconds": real_s, "steps_per_sec": sps,
            "max_rss_GB": gb(tm.get("max_rss_bytes")),
            "OMP_NUM_THREADS": meta.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": meta.get("MKL_NUM_THREADS"),
        })

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[methods-table] wrote {args.out} with {len(df)} rows")

if __name__ == "__main__":
    main()
