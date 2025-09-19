#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# --- [0/8] Preflight ---------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p data/CLENS data/CLENS/kids_tiles working outputs logs

echo "[0/8] Using repo at: $ROOT"

# --- [1/8] (Optional) Fetch KiDS tiles over W1 patch ------------------------
# Safe to re-run: existing files are kept; empty tiles are fine.
python - <<'PY'
import os, subprocess, pathlib
tiles_dir="data/CLENS/kids_tiles"
pathlib.Path(tiles_dir).mkdir(parents=True, exist_ok=True)

def fetch(ra1,ra2,dec1,dec2):
    url=("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?"
         "-source=II/384/kids_shear&"
         "-out=RAJ2000,DEJ2000,e1,e2,Weight,zbest,FitClass&"
         f"-oc.form=dec&RAJ2000={ra1:.3f}..{ra2:.3f}&DEJ2000={dec1:.3f}..{dec2:.3f}&"
         "-out.max=1000000")
    out=f"{tiles_dir}/kids_{ra1:.3f}_{ra2:.3f}_{dec1:.3f}_{dec2:.3f}.tsv"
    if os.path.exists(out):
        print("keep:", out); return
    print("fetch:", out)
    # -f: fail on server errors, but we allow empty tiles (the file will be tiny)
    subprocess.run(["curl","-fsSL","-o",out,url], check=False)

for ra1 in range(210, 221, 2):   # 210..220 in 2° stripes
    ra2=ra1+2
    for dec1 in range(-5, 5, 2): # -5..+3 in 2° stripes (top row often empty)
        dec2=dec1+2
        fetch(ra1,ra2,dec1,dec2)
PY

# --- [2/8] Merge tiles (skip empties) ---------------------------------------
python - <<'PY'
import pandas as pd, glob, os
from pandas.errors import EmptyDataError, ParserError
tiles=sorted(glob.glob("data/CLENS/kids_tiles/*.tsv"))
parts=[]
for t in tiles:
    try:
        df = pd.read_csv(t, sep="\t", comment="#", low_memory=False)
        if df.empty or df.shape[1]==0:
            print("[warn] skip", os.path.basename(t), "empty file")
            continue
        parts.append(df)
    except (EmptyDataError, ParserError):
        print("[warn] skip", os.path.basename(t), "No columns to parse from file")

if not parts:
    raise SystemExit("[error] No non-empty tiles found.")

df = pd.concat(parts, ignore_index=True)
df.drop_duplicates(subset=["RAJ2000","DEJ2000","e1","e2","Weight","zbest","FitClass"],
                   inplace=True)
out="data/CLENS/kids_shear_wide.tsv"
df.to_csv(out, sep="\t", index=False)
print(f"[merge] -> {out} rows:", len(df))
PY

# --- [3/8] Build sources.csv for the stacker --------------------------------
python - <<'PY'
import pandas as pd, pathlib
tsv = "data/CLENS/kids_shear_wide.tsv"
df  = pd.read_csv(tsv, sep="\t", comment="#", low_memory=False)
df.rename(columns={'RAJ2000':'ra','DEJ2000':'dec','Weight':'w',
                   'zbest':'z_src','FitClass':'fitclass'}, inplace=True)
# force numerics
for c in ['ra','dec','e1','e2','w','z_src','fitclass']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
# KiDS: galaxies only (fitclass==0) & w>0
df = df[(df['fitclass']==0) & (df['w']>0)].dropna(subset=['ra','dec','e1','e2','w','z_src'])
if 'm' not in df.columns:
    df['m']=0.0
out="data/CLENS/sources.csv"
df[['ra','dec','e1','e2','w','z_src','m']].to_csv(out, index=False)
print(f"[sources] -> {out} rows:", len(df))
PY

# --- [4/8] Build MCXC cluster patch that overlaps sources -------------------
# Uses your existing MCXC file if present; else stop with a helpful message.
python - <<'PY'
import pandas as pd, os, numpy as np
src = pd.read_csv("data/CLENS/sources.csv")
clp = "data/CLENS/clusters.csv"
if not os.path.exists(clp):
    raise SystemExit("[error] data/CLENS/clusters.csv not found. "
                     "Use the MCXC fetch step you ran earlier to create it.")
cl  = pd.read_csv(clp)
# Force numeric deg in case
for c in ['ra','dec','z_cl']:
    cl[c] = pd.to_numeric(cl[c], errors='coerce')
cl = cl.dropna(subset=['ra','dec','z_cl'])
ra_min,ra_max = src['ra'].min(), src['ra'].max()
dec_min,dec_max = src['dec'].min(), src['dec'].max()
in_patch = cl[(cl['ra']>=ra_min)&(cl['ra']<=ra_max)&(cl['dec']>=dec_min)&(cl['dec']<=dec_max)]
out="data/CLENS/clusters_patch.csv"
in_patch.to_csv(out, index=False)
print(f"[clusters] patch -> {out} (rows={len(in_patch)})")
PY

# --- [5/8] Prepare parquet for the stacker ----------------------------------
python -u scripts/clens_prepare.py \
  --clusters data/CLENS/clusters_patch.csv \
  --sources  data/CLENS/sources.csv \
  --outdir   working/CLENS_patch

# --- [6/8] Baseline stack (focus/control) -----------------------------------
python -u scripts/clens_stack.py \
  --work working/CLENS_patch \
  --out  outputs/CLENS_patch \
  --z_focus 0.12,0.16 \
  --z_ctrl  "0.06,0.10;0.18,0.30"

# --- [7/8] Nulls: 45°-rotated & stratified randoms --------------------------
# Rotated (e1,e2) -> (-e2, e1)
python - <<'PY'
import pandas as pd, shutil, pathlib, subprocess
base='working/CLENS_patch'; rot='working/CLENS_patch_rot'
pathlib.Path(rot).mkdir(parents=True, exist_ok=True)
shutil.copy(f'{base}/clusters.parquet', f'{rot}/clusters.parquet')
s=pd.read_parquet(f'{base}/sources.parquet'); s['e1'], s['e2'] = -s['e2'].values, s['e1'].values
s.to_parquet(f'{rot}/sources.parquet', index=False)
subprocess.run(['python','-u','scripts/clens_stack.py','--work',rot,'--out','outputs/CLENS_patch_rot',
                '--z_focus','0.12,0.16','--z_ctrl','0.06,0.10;0.18,0.30'], check=True)
print("[rot] stacked")
PY

# Stratified randoms: preserve the lenses' z_cl counts in each window
python - <<'PY'
import numpy as np, pandas as pd, pathlib, subprocess, json
rng=np.random.default_rng(5)
src=pd.read_parquet("working/CLENS_patch/sources.parquet")
cl =pd.read_parquet("working/CLENS_patch/clusters.parquet")
zf=(0.12,0.16); zc=[(0.06,0.10),(0.18,0.30)]
def counts(cl):
    nF=((cl['z_cl']>=zf[0])&(cl['z_cl']<=zf[1])).sum()
    nC=sum(((cl['z_cl']>=a)&(cl['z_cl']<=b)).sum() for a,b in zc)
    return nF,nC
nF,nC = counts(cl)
ra_min,ra_max=src['ra'].min(),src['ra'].max()
dec_min,dec_max=src['dec'].min(),src['dec'].max()
outs=[]
for k in range(1,31):  # 30 randoms with the same (nF,nC)
    rows=[]
    # focus clones
    for _ in range(nF):
        rows.append({'ra':rng.uniform(ra_min,ra_max),
                     'dec':rng.uniform(dec_min,dec_max),
                     'z_cl':rng.uniform(zf[0],zf[1])})
    # control clones
    for a,b in zc:
        m = ((cl['z_cl']>=a)&(cl['z_cl']<=b)).sum()
        for _ in range(int(m)):
            rows.append({'ra':rng.uniform(ra_min,ra_max),
                         'dec':rng.uniform(dec_min,dec_max),
                         'z_cl':rng.uniform(a,b)})
    rnd=pd.DataFrame(rows)
    d=f"data/CLENS/random_strat_{k:02d}.csv"; rnd.to_csv(d,index=False)
    w=f"working/CLENS_random_strat_{k:02d}"
    pathlib.Path(w).mkdir(parents=True,exist_ok=True)
    subprocess.run(["python","-u","scripts/clens_prepare.py","--clusters",d,"--sources",
                    "data/CLENS/sources.csv","--outdir",w], check=True)
    o=f"outputs/CLENS_random_strat_{k:02d}"
    subprocess.run(["python","-u","scripts/clens_stack.py","--work",w,"--out",o,
                    "--z_focus","0.12,0.16","--z_ctrl","0.06,0.10;0.18,0.30"], check=True)
    outs.append(o)
json.dump(outs, open("outputs/CLENS_random_strat_list.json","w"))
print("[random] stratified stacks:", len(outs))
PY

# --- [8/8] Figure + band-mean table -----------------------------------------
python - <<'PY'
import pandas as pd, numpy as np, matplotlib.pyplot as plt, json, os
def L(path):
    if not os.path.exists(path): return None
    d=pd.read_csv(path); r='R_Mpc' if 'R_Mpc' in d.columns else 'R'; g='gt' if 'gt' in d.columns else 'g_t'
    return d[[r,g]].rename(columns={r:'R_Mpc',g:'gt'})
def bandmean(path,lo=0.5,hi=1.8):
    d=pd.read_csv(path); r='R_Mpc' if 'R_Mpc' in d.columns else 'R'; g='gt' if 'gt' in d.columns else 'g_t'
    m=d[(d[r]>=lo)&(d[r]<=hi)][g].mean(); return float(m) if m==m else np.nan
curves = {
  'Focus'  : L('outputs/CLENS_patch/profile_focus.csv'),
  'Control': L('outputs/CLENS_patch/profile_control.csv'),
  'Rotated': L('outputs/CLENS_patch_rot/profile_focus.csv'),
}
outs=json.load(open('outputs/CLENS_random_strat_list.json'))
acc=None
for o in outs:
    d=L(f'{o}/profile_focus.csv')
    acc = d if acc is None else acc.add(d, fill_value=0.0)
rand = acc.copy(); rand['gt'] /= len(outs)
curves['Random']=rand
plt.figure(figsize=(7,5))
for k,v in curves.items():
    if v is not None: plt.plot(v['R_Mpc'], v['gt'], '-o', label=k)
plt.axvspan(0.5,1.8,alpha=0.12)
plt.xlabel('R [Mpc]'); plt.ylabel(r'$g_t$'); plt.legend()
os.makedirs('outputs/CLENS_release', exist_ok=True)
plt.tight_layout(); plt.savefig('outputs/CLENS_release/gt_panel.png', dpi=160)
summary=pd.DataFrame({
 'set':['Focus','Control','Random','Rotated'],
 'gt_mean_0p5_1p8':[bandmean('outputs/CLENS_patch/profile_focus.csv'),
                    bandmean('outputs/CLENS_patch/profile_control.csv'),
                    np.mean([bandmean(f"{o}/profile_focus.csv") for o in outs]),
                    bandmean('outputs/CLENS_patch_rot/profile_focus.csv')]
})
summary.to_csv('outputs/CLENS_release/band_means.csv', index=False)
print(summary)
print("[release] outputs/CLENS_release/gt_panel.png + band_means.csv")
PY

echo "[DONE] clens_end_to_end.sh"
