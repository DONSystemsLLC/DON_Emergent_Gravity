#!/usr/bin/env python3
"""
CLENS Plots - Enhanced for P2 Slip Prototype
Generates publication-quality lensing-dynamics comparison plots
"""
import glob
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = "outputs/CLENS_release"

def load_prof(p):
    """Load profile with flexible column naming"""
    if not os.path.exists(p):
        return None, None, None
    d = pd.read_csv(p)
    R = 'R_Mpc' if 'R_Mpc' in d.columns else 'R'
    g = 'gt' if 'gt' in d.columns else 'g_t'
    
    # Apply K-correction if available
    if 'K_1p' in d.columns and not d['K_1p'].isna().all():
        d[g] = d[g] / (1.0 + d['K_1p'])
    
    d = d.reset_index(drop=True)
    d['bin'] = d.index
    return d, R, g

def wband(d, R, g, lo, hi):
    """Weighted band average"""
    if d is None:
        return np.nan
    m = (d[R] >= lo) & (d[R] <= hi)
    y = d.loc[m, g].to_numpy()
    if y.size == 0:
        return np.nan
    
    # Use weights if available
    if 'W' in d.columns:
        w = d.loc[m, 'W'].to_numpy()
        m2 = np.isfinite(y) & np.isfinite(w) & (w > 0)
        return np.average(y[m2], weights=w[m2]) if m2.any() else np.nan
    return np.nanmean(y)

def compute_band_means(profiles, R_col, g_col, bands=None):
    """Compute band means for multiple radial bands"""
    if bands is None:
        bands = [(0.3, 0.6), (0.5, 1.0), (0.5, 1.8), (1.0, 2.0), (1.5, 2.5)]
    
    results = []
    for label, (rlo, rhi) in [('band1', bands[0]), ('band2', bands[1]), 
                              ('band3', bands[2]), ('band4', bands[3]), ('band5', bands[4])]:
        row = {'band': label, 'r_min': rlo, 'r_max': rhi}
        
        for key, profile in profiles.items():
            if profile[0] is not None:
                d, R, g = profile
                row[f'{key}_gt_mean'] = wband(d, R, g, rlo, rhi)
            else:
                row[f'{key}_gt_mean'] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)

def create_gt_panel(profiles, R_col, g_col, outdir):
    """Create publication-quality gt panel plot with enhanced styling for P2"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Enhanced color scheme and styling
    colors = {
        'Focus': '#d62728',      # Red
        'Control': '#1f77b4',    # Blue  
        'Random': '#2ca02c',     # Green
        'Rotated': '#9467bd'     # Purple
    }
    
    markers = {
        'Focus': 'o',
        'Control': 's', 
        'Random': '^',
        'Rotated': 'D'
    }
    
    # Plot profiles with enhanced styling
    for key, profile in profiles.items():
        if profile[0] is not None:
            d, R, g = profile
            color = colors.get(key, 'black')
            marker = markers.get(key, 'o')
            
            # Add error bars if available
            if f'{g}_err' in d.columns:
                ax.errorbar(d[R], d[g], yerr=d[f'{g}_err'], 
                           marker=marker, color=color, label=key,
                           markersize=6, linewidth=2, capsize=3)
            else:
                ax.plot(d[R], d[g], marker=marker, color=color, label=key,
                       markersize=6, linewidth=2, linestyle='-', alpha=0.9)
    
    # Enhanced band highlighting - key analysis region
    ax.axvspan(0.5, 1.8, alpha=0.12, color='gray', 
               label='Analysis band (0.5–1.8 Mpc)', zorder=0)
    
    # Additional reference bands
    ax.axvspan(0.3, 0.6, alpha=0.06, color='orange', 
               label='Inner band (0.3–0.6 Mpc)', zorder=0)
    ax.axvspan(1.5, 2.5, alpha=0.06, color='purple', 
               label='Outer band (1.5–2.5 Mpc)', zorder=0)
    
    # Zero reference line
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Enhanced styling
    ax.set_xlabel('R [Mpc]', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$g_t$', fontsize=14, fontweight='bold')
    ax.set_title('KiDS W1 × Cluster Weak Lensing Stack — P2 Slip Analysis', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Enhanced legend with better positioning
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                      shadow=True, fontsize=11)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Better axis formatting
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save with high quality
    output_path = os.path.join(outdir, 'gt_panel.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"[clens] Saved enhanced gt_panel: {output_path}")
    
    # Also save as PDF for publication
    pdf_path = os.path.join(outdir, 'gt_panel.pdf') 
    plt.savefig(pdf_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"[clens] Saved publication PDF: {pdf_path}")
    
    plt.close()
    return output_path

def main():
    """Main CLENS plotting routine"""
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    
    print("[clens] Loading CLENS profiles...")
    
    # Load main profiles
    profiles = {}
    profile_paths = {
        'Focus': 'outputs/CLENS_patch/profile_focus.csv',
        'Control': 'outputs/CLENS_patch/profile_control.csv',
        'Rotated': 'outputs/CLENS_patch_rot/profile_focus.csv'
    }
    
    for key, path in profile_paths.items():
        profiles[key] = load_prof(path)
        status = "✓" if profiles[key][0] is not None else "✗"
        print(f"  {key}: {status} {path}")
    
    # Collect and average random profiles
    print("[clens] Collecting random ensemble...")
    random_dirs = sorted({os.path.dirname(p) for p in glob.glob('outputs/CLENS_random_strat3_*/profile_*.csv')})
    
    rF_list = []
    for d in random_dirs:
        pf = os.path.join(d, 'profile_focus.csv')
        if os.path.exists(pf):
            df, _, _ = load_prof(pf)
            if df is not None:
                rF_list.append(df)
    
    if rF_list:
        print(f"[clens] Found {len(rF_list)} random realizations")
        # Average the random profiles
        concat_random = pd.concat(rF_list, ignore_index=True)
        random_avg = concat_random.groupby('bin', as_index=False).mean(numeric_only=True)
        
        # Create a synthetic random profile with consistent columns
        if 'Focus' in profiles and profiles['Focus'][0] is not None:
            focus_d, focus_R, focus_g = profiles['Focus']
            random_profile = focus_d.copy()
            if len(random_avg) == len(focus_d):
                random_profile[focus_g] = random_avg[focus_g]
            profiles['Random'] = (random_profile, focus_R, focus_g)
        else:
            profiles['Random'] = (None, None, None)
    else:
        print("[clens] No random realizations found")
        profiles['Random'] = (None, None, None)
    
    # Extract column names from first valid profile
    R_col = g_col = None
    for key, profile in profiles.items():
        if profile[0] is not None:
            _, R_col, g_col = profile
            break
    
    if R_col is None or g_col is None:
        print("[clens] Error: No valid profiles found")
        return
    
    print(f"[clens] Using columns: R={R_col}, g_t={g_col}")
    
    # Generate band means table
    print("[clens] Computing band means...")
    band_means = compute_band_means(profiles, R_col, g_col)
    
    # Add derived columns
    if 'Focus_gt_mean' in band_means.columns and 'Control_gt_mean' in band_means.columns:
        band_means['Delta_gt'] = band_means['Focus_gt_mean'] - band_means['Control_gt_mean']
    
    if 'Random_gt_mean' in band_means.columns:
        band_means['Signal_over_Random'] = band_means.get('Delta_gt', np.nan) / band_means['Random_gt_mean']
    
    # Save band means
    band_means_path = os.path.join(OUTDIR, 'band_means.csv')
    band_means.to_csv(band_means_path, index=False)
    print(f"[clens] Saved {band_means_path}")
    print(band_means.to_string(index=False))
    
    # Create gt panel plot
    print("[clens] Creating gt panel...")
    gt_panel_path = create_gt_panel(profiles, R_col, g_col, OUTDIR)
    
    # Additional legacy plots
    print("[clens] Creating additional diagnostic plots...")
    
    # 1) Random-Δ histogram in 0.5–1.8 Mpc (if data available)
    if profiles['Focus'][0] is not None and profiles['Control'][0] is not None and rF_list:
        rlo, rhi = 0.5, 1.8
        d_rand = []
        
        # Recompute for each random realization
        random_dirs = sorted({os.path.dirname(p) for p in glob.glob('outputs/CLENS_random_strat3_*/profile_*.csv')})
        for d in random_dirs:
            pf = os.path.join(d, 'profile_focus.csv')
            pc = os.path.join(d, 'profile_control.csv')
            if os.path.exists(pf) and os.path.exists(pc):
                df, R_f, g_f = load_prof(pf)
                dc, R_c, g_c = load_prof(pc)
                if df is not None and dc is not None:
                    delta = wband(df, R_f, g_f, rlo, rhi) - wband(dc, R_c, g_c, rlo, rhi)
                    if np.isfinite(delta):
                        d_rand.append(delta)
        
        if d_rand:
            plt.figure(figsize=(8, 5))
            plt.hist(d_rand, bins=20, edgecolor='k', alpha=0.7)
            plt.axvline(np.nanmean(d_rand), lw=2, color='red', label=f'Mean: {np.nanmean(d_rand):.4f}')
            plt.title(f"Random Δ g_t (focus-control), {rlo}-{rhi} Mpc (N={len(d_rand)})")
            plt.xlabel("Δ g_t")
            plt.ylabel("Count")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            hist_path = os.path.join(OUTDIR, "random_delta_hist.png")
            plt.savefig(hist_path, dpi=150, bbox_inches='tight')
            print(f"[clens] Saved {hist_path}")
    
    # 2) Boost factor (if N_src column available)
    if profiles['Focus'][0] is not None and 'N_src' in profiles['Focus'][0].columns and rF_list:
        dF = profiles['Focus'][0]
        if len(rF_list) > 0:
            concat = pd.concat(rF_list, ignore_index=True)
            rmean = concat.groupby('bin', as_index=False).mean(numeric_only=True)
            
            if 'N_src' in rmean.columns:
                B = dF['N_src'].to_numpy() / np.where(rmean['N_src'].to_numpy() > 0, rmean['N_src'].to_numpy(), np.nan)
                
                plt.figure(figsize=(8, 5))
                plt.plot(dF[R_col].to_numpy(), B, marker='o', linestyle='-')
                plt.xlabel(f'{R_col}')
                plt.ylabel("B(R)")
                plt.title("Boost factor B(R) = N_lens / <N_rand>")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                boost_path = os.path.join(OUTDIR, "boost_B_of_R.png")
                plt.savefig(boost_path, dpi=150, bbox_inches='tight')
                print(f"[clens] Saved {boost_path}")
    
    # Summary JSON
    summary = {
        'profiles_loaded': {k: v[0] is not None for k, v in profiles.items()},
        'n_random_realizations': len(rF_list),
        'output_files': {
            'band_means_csv': band_means_path,
            'gt_panel_png': gt_panel_path
        },
        'columns_used': {'R': R_col, 'g_t': g_col}
    }
    
    summary_path = os.path.join(OUTDIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[clens] Saved summary to {summary_path}")
    print(f"[clens] CLENS plots complete. Key outputs in {OUTDIR}/")

if __name__ == "__main__":
    main()