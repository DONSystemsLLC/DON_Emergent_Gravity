#!/usr/bin/env python3
"""
Helmholtz Diagnostics - Enhanced for P2 with flux flatness analysis
Analyzes curl-free nature and flux constancy in the fit annulus
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def curl_components(Fx, Fy, Fz, dx):
    # central diff with PBC
    roll = lambda a, s, ax: np.roll(a, s, axis=ax)
    dFz_dy = (roll(Fz, -1, 1) - roll(Fz, 1, 1)) / (2*dx)
    dFy_dz = (roll(Fy, -1, 2) - roll(Fy, 1, 2)) / (2*dx)
    dFx_dz = (roll(Fx, -1, 2) - roll(Fx, 1, 2)) / (2*dx)
    dFz_dx = (roll(Fz, -1, 0) - roll(Fz, 1, 0)) / (2*dx)
    dFy_dx = (roll(Fy, -1, 0) - roll(Fy, 1, 0)) / (2*dx)
    dFx_dy = (roll(Fx, -1, 1) - roll(Fx, 1, 1)) / (2*dx)
    cx = dFz_dy - dFy_dz
    cy = dFx_dz - dFz_dx
    cz = dFy_dx - dFx_dy
    return cx, cy, cz

def load_J(npz_path):
    z = np.load(npz_path)
    # Common key aliases
    cand = [
        ("Jx","Jy","Jz"),
        ("Fx","Fy","Fz"),
        ("gradPhi_x","gradPhi_y","gradPhi_z"),
        ("ax","ay","az"),  # acceleration field
        ("gx","gy","gz"),
    ]
    for kx,ky,kz in cand:
        if kx in z and ky in z and kz in z:
            return z[kx].astype(np.float64), z[ky].astype(np.float64), z[kz].astype(np.float64)
    raise KeyError("No vector field found. Expected one of (Jx,Jy,Jz)/(Fx,Fy,Fz)/…")

def shell_average(r, vals, rmin, rmax, nbins=48):
    edges = np.linspace(rmin, rmax, nbins+1)
    idx = np.digitize(r, edges)-1
    out = []
    for i in range(nbins):
        m = idx==i
        if not np.any(m): continue
        rmid = 0.5*(edges[i]+edges[i+1])
        out.append((rmid, np.mean(vals[m]), np.median(vals[m]), np.std(vals[m]), len(vals[m])))
    return np.array(out)

def compute_flux_profile(r, Jr, rmin, rmax, nbins=48):
    """Compute flux F(r) = <|Jr|> * 4πr² for flatness analysis"""
    edges = np.linspace(rmin, rmax, nbins+1)
    idx = np.digitize(r, edges)-1
    flux_profile = []
    
    for i in range(nbins):
        m = idx == i
        if not np.any(m): 
            continue
            
        rmid = 0.5*(edges[i] + edges[i+1])
        Jr_mean = np.mean(np.abs(Jr[m]))
        Jr_med = np.median(np.abs(Jr[m]))
        flux_mean = Jr_mean * 4 * np.pi * rmid**2
        flux_med = Jr_med * 4 * np.pi * rmid**2
        
        flux_profile.append((rmid, Jr_mean, Jr_med, flux_mean, flux_med, len(Jr[m])))
    
    return np.array(flux_profile)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field_npz", required=True, help="NPZ snapshot with vector field")
    ap.add_argument("--L", type=float, required=True, help="box length")
    ap.add_argument("--rmin", type=float, default=12.0)
    ap.add_argument("--rmax", type=float, default=32.0)
    ap.add_argument("--out_csv", default="figs/helmholtz_checks.csv")
    ap.add_argument("--out_json", default="figs/helmholtz_checks.json")
    args = ap.parse_args()

    print(f"[helmholtz] Loading field from {args.field_npz}")
    Jx, Jy, Jz = load_J(args.field_npz)
    N = Jx.shape[0]
    dx = args.L / N

    print(f"[helmholtz] Grid: N={N}, L={args.L}, dx={dx:.4f}")

    # Coordinates centered at box center
    ax = np.arange(N) - N/2
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    X = (X*dx); Y = (Y*dx); Z = (Z*dx)
    r = np.sqrt(X**2 + Y**2 + Z**2) + 1e-12
    rhat = np.stack([X/r, Y/r, Z/r], axis=-1)

    # |J| and tangential fraction
    J = np.stack([Jx,Jy,Jz], axis=-1)
    Jmag = np.linalg.norm(J, axis=-1) + 1e-18
    Jr = (J[...,0]*rhat[...,0] + J[...,1]*rhat[...,1] + J[...,2]*rhat[...,2])
    Jtan_mag = np.sqrt(np.clip(Jmag**2 - Jr**2, 0.0, None))
    tau_frac = (Jtan_mag / Jmag)

    # curl(J)
    cx, cy, cz = curl_components(Jx,Jy,Jz,dx)
    curlJ = np.sqrt(cx*cx + cy*cy + cz*cz)

    # shell averages in [rmin,rmax]
    sel = (r>=args.rmin) & (r<=args.rmax)
    r_sel = r[sel]
    Rcurl = (curlJ/Jmag)[sel]
    Tau = tau_frac[sel]
    RadErr = 1.0 - np.abs(Jr[sel])/Jmag[sel]
    Jr_sel = Jr[sel]

    print(f"[helmholtz] Analyzing {np.sum(sel)} points in annulus [{args.rmin}, {args.rmax}]")

    A1 = shell_average(r_sel, Rcurl, args.rmin, args.rmax)
    A2 = shell_average(r_sel, Tau,   args.rmin, args.rmax)
    A3 = shell_average(r_sel, RadErr,args.rmin, args.rmax)

    # NEW: Flux flatness analysis
    flux_profile = compute_flux_profile(r_sel, Jr_sel, args.rmin, args.rmax)
    
    if len(flux_profile) > 0:
        # Flux flatness: relative variation of F(r) = <|Jr|> * 4πr²
        flux_vals = flux_profile[:, 4]  # median flux
        flux_med = np.median(flux_vals)
        flux_rel = flux_vals / flux_med if flux_med > 0 else flux_vals
        flux_p16, flux_p84 = np.percentile(flux_rel, [16, 84])
        flux_p5, flux_p95 = np.percentile(flux_rel, [5, 95])
        flux_flatness_p95 = max(abs(flux_p95 - 1.0), abs(flux_p5 - 1.0))
        
        print(f"[helmholtz] Flux flatness (p5-p95): {flux_p5:.3f} - {flux_p95:.3f}")
        print(f"[helmholtz] Flux flatness (p95 deviation): {flux_flatness_p95:.3f}")

    # write CSV
    df_data = {
        "r_mid": A1[:,0],
        "Rcurl_mean": A1[:,1], "Rcurl_median": A1[:,2], "Rcurl_std": A1[:,3], "Rcurl_count": A1[:,4],
        "tau_frac_mean": A2[:,1], "tau_frac_median": A2[:,2], "tau_frac_std": A2[:,3], "tau_frac_count": A2[:,4],
        "radiality_err_mean": A3[:,1], "radiality_err_median": A3[:,2], "radiality_err_std": A3[:,3], "radiality_err_count": A3[:,4]
    }
    
    # Add flux profile if available
    if len(flux_profile) > 0 and len(flux_profile) == len(A1):
        df_data.update({
            "Jr_mean": flux_profile[:,1],
            "Jr_median": flux_profile[:,2], 
            "flux_mean": flux_profile[:,3],
            "flux_median": flux_profile[:,4],
            "flux_count": flux_profile[:,5]
        })

    df = pd.DataFrame(df_data)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[helmholtz] Wrote {args.out_csv}")

    # Enhanced summary with acceptance criteria
    summary = {
        "window": [args.rmin, args.rmax],
        "grid_info": {"N": int(N), "L": float(args.L), "dx": float(dx)},
        "n_points_analyzed": int(np.sum(sel)),
        "curl_diagnostics": {
            "Rcurl_mean": float(np.mean(Rcurl)),
            "Rcurl_median": float(np.median(Rcurl)),
            "Rcurl_p95": float(np.percentile(Rcurl, 95))
        },
        "tangential_diagnostics": {
            "tau_frac_mean": float(np.mean(Tau)),
            "tau_frac_median": float(np.median(Tau)),
            "tau_frac_p95": float(np.percentile(Tau, 95))
        },
        "radiality_diagnostics": {
            "radiality_err_mean": float(np.mean(RadErr)),
            "radiality_err_median": float(np.median(RadErr)),
            "radiality_err_p95": float(np.percentile(RadErr, 95))
        }
    }
    
    # Add flux diagnostics
    if len(flux_profile) > 0:
        summary["flux_diagnostics"] = {
            "flux_median": float(flux_med),
            "flux_flatness_p16_p84": [float(flux_p16), float(flux_p84)],
            "flux_flatness_p5_p95": [float(flux_p5), float(flux_p95)],
            "flux_flatness_p95_dev": float(flux_flatness_p95)
        }
        
        # P2 acceptance criteria (tightened)
        summary["p2_acceptance"] = {
            "Rcurl_ok": float(np.mean(Rcurl)) < 0.05,           # Rcurl < 0.05
            "tau_frac_ok": float(np.mean(Tau)) < 0.15,          # τ_frac < 0.15  
            "radiality_err_ok": float(np.mean(RadErr)) < 0.10,  # radiality_err < 0.10
            "flux_flatness_ok": flux_flatness_p95 <= 0.05       # flux p95 ≤ 0.05
        }
        
        n_pass = sum(summary["p2_acceptance"].values())
        summary["p2_acceptance"]["score"] = f"{n_pass}/4"
        summary["p2_acceptance"]["all_pass"] = n_pass == 4
    else:
        summary["flux_diagnostics"] = {"error": "No flux profile computed"}
        summary["p2_acceptance"] = {"error": "Missing flux data"}

    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(f"[helmholtz] Wrote {args.out_json}")
    
    # Print summary  
    print(f"[helmholtz] Summary:")
    print(f"  Rcurl mean: {summary['curl_diagnostics']['Rcurl_mean']:.3e} (target: <0.05)")
    print(f"  tau_frac mean: {summary['tangential_diagnostics']['tau_frac_mean']:.3e} (target: <0.15)")
    print(f"  radiality_err mean: {summary['radiality_diagnostics']['radiality_err_mean']:.3e} (target: <0.10)")
    
    if "flux_diagnostics" in summary and "flux_flatness_p95_dev" in summary["flux_diagnostics"]:
        print(f"  flux flatness p95: {summary['flux_diagnostics']['flux_flatness_p95_dev']:.3e} (target: ≤0.05)")
        
        if "p2_acceptance" in summary and "all_pass" in summary["p2_acceptance"]:
            status = "✓ PASS" if summary["p2_acceptance"]["all_pass"] else "✗ FAIL"
            score = summary["p2_acceptance"]["score"]
            print(f"  P2 Acceptance: {status} ({score})")

if __name__ == "__main__":
    main()