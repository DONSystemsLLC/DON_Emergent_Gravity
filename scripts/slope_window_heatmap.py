#!/usr/bin/env python3
"""
Slope Window Heatmap - Generate window-independence figure for P2
Shows slope stability across different radial analysis windows with Theil-Sen robust fitting
"""
import argparse, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
try:
    from sklearn.linear_model import TheilSenRegressor
    from sklearn.utils import resample
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

def fit_theilsen(x, y, n_boot=200):
    """Fit Theil-Sen robust regression with bootstrap confidence intervals"""
    if not HAS_SKLEARN:
        # Fallback to simple linear regression
        p = np.polyfit(np.log(x), np.log(np.abs(y)), 1)[0]
        return p, (p-0.05, p+0.05)  # crude CI
        
    x_log = np.log(x).reshape(-1,1)
    y_log = np.log(np.abs(y))
    
    # Main fit
    model = TheilSenRegressor(random_state=0).fit(x_log, y_log)
    p = float(model.coef_[0])
    
    # Bootstrap confidence interval
    boots = []
    for _ in range(n_boot):
        xb, yb = resample(x_log, y_log, replace=True, random_state=None)
        try:
            boots.append(TheilSenRegressor(random_state=0).fit(xb, yb).coef_[0])
        except:
            continue
    
    if boots:
        lo, hi = np.percentile(boots, [16, 84])
    else:
        lo, hi = p-0.05, p+0.05
        
    return p, (float(lo), float(hi))

def load_field_and_compute_profile(field_npz, L, rmin=6.0, rmax=40.0, Nr=48):
    """Load NPZ field and compute radial profile"""
    z = np.load(field_npz)
    
    # Try common field names
    for jx_key in ['Jx', 'Fx', 'gradPhi_x', 'ax']:
        if jx_key in z:
            Jx = z[jx_key]
            Jy = z[jx_key.replace('x', 'y')]
            Jz = z[jx_key.replace('x', 'z')]
            break
    else:
        raise KeyError("No vector field found in NPZ")
    
    N = Jx.shape[0]
    dx = L / N
    
    # Coordinates from box center
    ax = np.arange(N) - N//2
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    X = X * dx; Y = Y * dx; Z = Z * dx
    r = np.sqrt(X**2 + Y**2 + Z**2) + 1e-12
    
    # Radial component
    rhat_x = X / r; rhat_y = Y / r; rhat_z = Z / r
    Jr = Jx * rhat_x + Jy * rhat_y + Jz * rhat_z
    
    # Shell averaging
    r_edges = np.logspace(np.log10(rmin), np.log10(rmax), Nr+1)
    r_mids = np.sqrt(r_edges[:-1] * r_edges[1:])
    Jr_shells = []
    
    for i in range(Nr):
        mask = (r >= r_edges[i]) & (r < r_edges[i+1])
        if np.any(mask):
            Jr_shells.append(np.median(np.abs(Jr[mask])))
        else:
            Jr_shells.append(np.nan)
    
    return r_mids, np.array(Jr_shells)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field_npz", help="NPZ field file")
    ap.add_argument("--profile_csv", help="CSV with r,Fr columns (alternative to --field_npz)")
    ap.add_argument("--L", type=float, default=160.0, help="Box size (for NPZ mode)")
    ap.add_argument("--windows", default="6,20;8,24;10,28;12,32;14,36;16,40")
    ap.add_argument("--out_png", default="figs/slope_window_heatmap.png")
    ap.add_argument("--out_json", default="figs/slope_window_heatmap.json")
    args = ap.parse_args()

    # Load data
    if args.field_npz:
        print(f"[window] Computing profile from {args.field_npz}")
        r, Fr = load_field_and_compute_profile(args.field_npz, args.L)
    elif args.profile_csv:
        print(f"[window] Loading profile from {args.profile_csv}")
        df = pd.read_csv(args.profile_csv)
        r = df["r"].to_numpy() if "r" in df.columns else df.iloc[:,0].to_numpy()
        # Try multiple column names
        for_col = None
        for cand in ("Fr_med","Fr_mean","a_r","gradPhi_r","g_r","Jr_med","force_r"):
            if cand in df.columns: 
                for_col = cand
                break
        if for_col is None:
            raise SystemExit("No force/accel column found")
        Fr = df[for_col].to_numpy()
    else:
        raise SystemExit("Must provide either --field_npz or --profile_csv")

    # Parse windows
    windows = []
    for w in args.windows.split(";"):
        parts = w.split(",")
        if len(parts) == 2:
            windows.append([float(parts[0]), float(parts[1])])

    print(f"[window] Testing {len(windows)} analysis windows")
    
    # Fit slopes for each window
    results = []
    slopes = []
    cis = []
    
    for i, (rmin, rmax) in enumerate(windows):
        mask = (r >= rmin) & (r <= rmax) & np.isfinite(Fr) & (Fr > 0)
        if np.sum(mask) < 5:
            print(f"Warning: Window [{rmin},{rmax}] has <5 points, skipping")
            slopes.append(np.nan)
            cis.append((np.nan, np.nan))
            continue
            
        r_win = r[mask]
        Fr_win = Fr[mask]
        
        slope, (ci_lo, ci_hi) = fit_theilsen(r_win, Fr_win)
        slopes.append(slope)
        cis.append((ci_lo, ci_hi))
        
        # Store detailed results
        results.append({
            'window': f"[{rmin},{rmax}]",
            'rmin': rmin,
            'rmax': rmax,
            'n_points': int(np.sum(mask)),
            'slope': slope,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'ci_width': ci_hi - ci_lo
        })
        
        print(f"  [{rmin:4.1f},{rmax:4.1f}]: slope={slope:+.3f} CI=[{ci_lo:+.3f},{ci_hi:+.3f}] n={np.sum(mask)}")

    # Create heatmap/plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    # Top panel: slopes vs window
    x_pos = np.arange(len(windows))
    slopes_clean = [s for s in slopes if not np.isnan(s)]
    cis_clean = [c for c in cis if not any(np.isnan(c))]
    
    if slopes_clean:
        ax1.errorbar(x_pos, slopes, yerr=[[s-c[0] for s,c in zip(slopes,cis)], 
                                         [c[1]-s for s,c in zip(slopes,cis)]], 
                    fmt='o-', capsize=3)
        ax1.axhline(-2.0, ls='--', color='red', alpha=0.7, label='Target: -2.0')
        ax1.axhspan(-2.05, -1.95, alpha=0.2, color='green', label='±0.05 tolerance')
        ax1.set_ylabel('Slope')
        ax1.set_title('Window Independence Test (Theil-Sen Robust Fit)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Check acceptance criteria
        slope_range = max(slopes_clean) - min(slopes_clean) if len(slopes_clean) > 1 else 0
        mean_slope = np.mean(slopes_clean)
        max_ci_width = max([c[1]-c[0] for c in cis_clean]) if cis_clean else 0
        
        print(f"\n[window] Analysis:")
        print(f"  Slope range: {slope_range:.4f} (target: ≤0.10)")  
        print(f"  Mean slope: {mean_slope:.3f} (target: -2.00±0.05)")
        print(f"  Max CI width: {max_ci_width:.4f} (target: ≤0.05)")
        
        # Acceptance
        range_ok = slope_range <= 0.10
        mean_ok = abs(mean_slope + 2.0) <= 0.05
        ci_ok = max_ci_width <= 0.05 if cis_clean else False
        
        print(f"  ✓ Range stable: {range_ok}")
        print(f"  ✓ Mean accurate: {mean_ok}")
        print(f"  ✓ Confidence intervals tight: {ci_ok}")
    
    # Bottom panel: CI widths
    ci_widths = [c[1]-c[0] for c in cis]
    ax2.bar(x_pos, ci_widths, alpha=0.7)
    ax2.axhline(0.05, ls='--', color='red', alpha=0.7, label='CI target: ≤0.05')
    ax2.set_ylabel('CI Width')
    ax2.set_xlabel('Analysis Window')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Common x-axis formatting
    window_labels = [f"[{w[0]:.0f},{w[1]:.0f}]" for w in windows]
    for ax in [ax1, ax2]:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(window_labels, rotation=45)
    
    plt.tight_layout()
    
    # Save files
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=160, bbox_inches='tight')
    print(f"[window] Saved {args.out_png}")
    
    # JSON summary
    summary = {
        "field_source": args.field_npz or args.profile_csv,
        "n_windows": len(windows),
        "windows": [[w[0], w[1]] for w in windows],
        "results": results,
        "summary_stats": {
            "slope_range": float(max(slopes_clean) - min(slopes_clean)) if len(slopes_clean) > 1 else 0.0,
            "mean_slope": float(np.mean(slopes_clean)) if slopes_clean else np.nan,
            "max_ci_width": float(max([c[1]-c[0] for c in cis_clean])) if cis_clean else np.nan,
            "n_valid_windows": len(slopes_clean)
        },
        "acceptance": {
            "range_stable": bool(slope_range <= 0.10) if 'slope_range' in locals() else False,
            "mean_accurate": bool(abs(mean_slope + 2.0) <= 0.05) if 'mean_slope' in locals() else False,
            "ci_tight": bool(max_ci_width <= 0.05) if 'max_ci_width' in locals() and cis_clean else False
        }
    }
    
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[window] Saved {args.out_json}")

if __name__ == "__main__":
    main()

