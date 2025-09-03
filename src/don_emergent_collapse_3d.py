#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
don_emergent_collapse_3d.py

Emergence test for DON collapse memory in 3D:

  ∂t μ + ∇·J = S(x,t)                      (continuity of memory)
  τ ∂t J + J = - c^2 ∇μ                    (local transport; telegraph-like)

No Poisson, no 1/r^2 coded anywhere.

Readout (attractive & conservative):
  a(x,t) = + κ J(x,t) = - κ c^2 ∇μ  = - ∇Φ_em,   Φ_em = κ c^2 μ

Outputs
-------
- field_slope_3d.png      (log|J| vs log r, fitted slope)
- orbit_emergent_3d.png   (test particle under a=+κJ or a=-∇(κ c^2 μ); XY projection with seam-aware breaks)
- energy_emergent_3d.png  (E drift with Φ_em)
- angmom_emergent_3d.png  (|L| drift)

Examples
--------
# Measure slope (single source at box center)
python don_emergent_collapse_3d.py --test slope --N 128 --L 64 \
  --masses "32,32,32,1.0" --eta 1.0 --kappa 1.0 --tau 0.2 --c 1.0 \
  --sigma 1.0 --warm 4000 --dt 0.02 --rmin 4 --rmax 20

# Emergent orbit (freeze steady field)
python don_emergent_collapse_3d.py --test orbit --N 128 --L 64 \
  --masses "32,32,32,1.0" --eta 1.0 --kappa 1.0 --tau 0.2 --c 1.0 \
  --sigma 1.0 --warm 5000 --dt 0.02 --r0 10.0 --vtheta 0.32 --steps 8000
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os, sys, json, time, shlex, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# --------------------------- 3D collapse field ---------------------------

class CollapseField3D:
    """
    Periodic box [0,L]^3, uniform cubic grid N^3.

    μ: memory density
    J: memory flux (Jx,Jy,Jz)

    Update (semi-implicit in J-decay):
      J_{n+1} = (J_n + (dt/τ)*(-c^2 ∇μ_n)) / (1 + dt/τ)
      μ_{n+1} = μ_n + dt * ( -∇·J_{n+1} + S_n )
    """
    def __init__(self, N=128, L=64.0, tau=0.2, c=1.0, eta=1.0, kappa=1.0, sigma=1.0):
        self.N  = int(N)
        self.L  = float(L)
        self.dx = self.L / self.N
        self.tau   = float(tau)
        self.c     = float(c)
        self.eta   = float(eta)
        self.kappa = float(kappa)
        self.sigma = float(sigma)  # kernel width in *grid* units

        self.x = np.linspace(0, self.L, self.N, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.x, self.x, indexing='ij')

        self.mu = np.zeros((self.N,self.N,self.N), dtype=np.float64)
        self.Jx = np.zeros_like(self.mu)
        self.Jy = np.zeros_like(self.mu)
        self.Jz = np.zeros_like(self.mu)

        # 3D Gaussian kernel normalized to unit integral on the torus
        r2 = ((self.X - self.L/2)**2 + (self.Y - self.L/2)**2 + (self.Z - self.L/2)**2)
        s2 = (self.sigma*self.dx)**2
        g  = np.exp(-0.5*r2/s2)
        self.kernel = g / (g.sum()*self.dx**3)

        self.masses = []  # list of (x,y,z,m)

    def set_masses(self, quads):
        self.masses = [(float(x)%self.L, float(y)%self.L, float(z)%self.L, float(m))
                       for (x,y,z,m) in quads]

    # periodic shifts
    def _shift(self, A, i,j,k):
        return np.roll(np.roll(np.roll(A, i, axis=0), j, axis=1), k, axis=2)

    # ∇μ (centered FD)
    def grad_mu(self):
        dμdx = (self._shift(self.mu,-1,0,0) - self._shift(self.mu,+1,0,0)) / (2*self.dx)
        dμdy = (self._shift(self.mu,0,-1,0) - self._shift(self.mu,0,+1,0)) / (2*self.dx)
        dμdz = (self._shift(self.mu,0,0,-1) - self._shift(self.mu,0,0,+1)) / (2*self.dx)
        return dμdx, dμdy, dμdz

    # Spectral ∇μ (smoother for conservative readout)
    def spectral_grad_mu(self):
        k = 2*np.pi*np.fft.fftfreq(self.N, d=self.dx)
        KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
        MU_k = np.fft.fftn(self.mu)
        gx = np.fft.ifftn(1j*KX*MU_k).real
        gy = np.fft.ifftn(1j*KY*MU_k).real
        gz = np.fft.ifftn(1j*KZ*MU_k).real
        return gx, gy, gz

    # ∇·J
    def div_J(self):
        dJxdx = (self._shift(self.Jx,-1,0,0) - self._shift(self.Jx,+1,0,0)) / (2*self.dx)
        dJydy = (self._shift(self.Jy,0,-1,0) - self._shift(self.Jy,0,+1,0)) / (2*self.dx)
        dJzdz = (self._shift(self.Jz,0,0,-1) - self._shift(self.Jz,0,0,+1)) / (2*self.dx)
        return dJxdx + dJydy + dJzdz

    # S deposition
    def deposit(self):
        S = np.zeros_like(self.mu)
        for (xi, yi, zi, m) in self.masses:
            lam = self.eta * m
            i0 = int(np.floor(xi / self.dx)) % self.N
            j0 = int(np.floor(yi / self.dx)) % self.N
            k0 = int(np.floor(zi / self.dx)) % self.N
            g = np.roll(np.roll(np.roll(self.kernel,
                                        i0 - self.N//2, axis=0),
                                        j0 - self.N//2, axis=1),
                                        k0 - self.N//2, axis=2)
            S += lam * g
        return S

    # one PDE step
    def step(self, dt):
        # semi-implicit J decay
        dμdx, dμdy, dμdz = self.grad_mu()
        Ax = - self.c*self.c * dμdx
        Ay = - self.c*self.c * dμdy
        Az = - self.c*self.c * dμdz
        fac = 1.0 / (1.0 + dt/self.tau)
        self.Jx = fac * (self.Jx + (dt/self.tau)*Ax)
        self.Jy = fac * (self.Jy + (dt/self.tau)*Ay)
        self.Jz = fac * (self.Jz + (dt/self.tau)*Az)

        # continuity
        S = self.deposit()
        divJ = self.div_J()
        self.mu = self.mu + dt * (-divJ + S)

        # hygiene
        self.mu = np.nan_to_num(self.mu, nan=0.0, posinf=0.0, neginf=0.0)
        self.Jx = np.nan_to_num(self.Jx, nan=0.0, posinf=0.0, neginf=0.0)
        self.Jy = np.nan_to_num(self.Jy, nan=0.0, posinf=0.0, neginf=0.0)
        self.Jz = np.nan_to_num(self.Jz, nan=0.0, posinf=0.0, neginf=0.0)

    # -------------------- samplers --------------------

    def _frac_index(self, x):
        fx = (x/self.dx) % self.N
        i0 = np.floor(fx).astype(int) % self.N
        i1 = (i0+1) % self.N
        t  = fx - np.floor(fx)
        return i0, i1, t

    def sample_array(self, F, x, y, z):
        """Periodic trilinear sampling of a 3D array F at (x,y,z)."""
        i0,i1,tx = self._frac_index(x)
        j0,j1,ty = self._frac_index(y)
        k0,k1,tz = self._frac_index(z)
        F000=F[i0,j0,k0]; F100=F[i1,j0,k0]; F010=F[i0,j1,k0]; F110=F[i1,j1,k0]
        F001=F[i0,j0,k1]; F101=F[i1,j0,k1]; F011=F[i0,j1,k1]; F111=F[i1,j1,k1]
        F00 = (1-tx)*F000 + tx*F100
        F01 = (1-tx)*F001 + tx*F101
        F10 = (1-tx)*F010 + tx*F110
        F11 = (1-tx)*F011 + tx*F111
        F0 = (1-ty)*F00 + ty*F10
        F1 = (1-ty)*F01 + ty*F11
        return float((1-tz)*F0 + tz*F1)

    def sample_J(self, x, y, z):
        Jx = self.sample_array(self.Jx, x,y,z)
        Jy = self.sample_array(self.Jy, x,y,z)
        Jz = self.sample_array(self.Jz, x,y,z)
        return Jx, Jy, Jz

    def sample_mu(self, x, y, z):
        return self.sample_array(self.mu, x,y,z)

# --------------------------- Helmholtz projection ---------------------------

def project_curl_free_J(field):
    """
    One-shot Helmholtz projection of J onto its gradient (curl-free) part in k-space.
    Does NOT change dynamics; call once after warmup and before any readout.
    """
    Jx_k = np.fft.fftn(field.Jx)
    Jy_k = np.fft.fftn(field.Jy)
    Jz_k = np.fft.fftn(field.Jz)

    k = 2*np.pi*np.fft.fftfreq(field.N, d=field.dx)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    K2 = KX*KX + KY*KY + KZ*KZ

    dot = KX*Jx_k + KY*Jy_k + KZ*Jz_k
    with np.errstate(divide='ignore', invalid='ignore'):
        fac = np.where(K2>0.0, dot/K2, 0.0)

    field.Jx = np.fft.ifftn(KX*fac).real
    field.Jy = np.fft.ifftn(KY*fac).real
    field.Jz = np.fft.ifftn(KZ*fac).real

# --------------------------- slope measurement ---------------------------

def measure_slope_J(field, rmin, rmax, Nr=50, cx=None, cy=None, cz=None):
    """Sample |J| along +x from the source center; fit log|J| vs log r."""
    L = field.L
    if cx is None or cy is None or cz is None:
        cx = cy = cz = L/2.0
    rr = np.logspace(np.log10(rmin), np.log10(rmax), Nr)
    vals = []
    for r in rr:
        x = (cx + r) % L
        y = cy % L
        z = cz % L
        Jx,Jy,Jz = field.sample_J(x,y,z)
        vals.append(np.sqrt(Jx*Jx + Jy*Jy + Jz*Jz))
    rr = np.asarray(rr); vals = np.asarray(vals) + 1e-300
    p = np.polyfit(np.log(rr), np.log(vals), 1)[0]
    plt.figure(figsize=(5.6,4.4), dpi=140)
    plt.loglog(rr, vals, 'o-', label='|J|(r) (measured)')
    plt.loglog(rr, rr**(-2), '--', label='r^-2 (ref, up to scale)')
    plt.xlabel("r"); plt.ylabel("|J|"); plt.legend()
    plt.title(f"3D log|J| slope ≈ {p:.3f} (target -2.000)")
    plt.tight_layout(); plt.savefig("field_slope_3d.png"); plt.close()
    print(f"[slope_3d] fitted slope p ≈ {p:.6f}  (target -2.000)")
    return float(p)

def test_slope_3d(N=128, L=64.0, masses=[(32,32,32,1.0)], eta=1.0, kappa=1.0, tau=0.2, c=1.0,
                  dt=0.02, warm=5000, rmin=4.0, rmax=20.0, sigma=1.0, project_curl_free=False):
    fld = CollapseField3D(N=N, L=L, tau=tau, c=c, eta=eta, kappa=kappa, sigma=sigma)
    fld.set_masses(masses)
    for _ in range(int(warm)):
        fld.step(dt)
    if project_curl_free:
        project_curl_free_J(fld)
        print("[curl_free] J projected before slope fit")
    cx,cy,cz,_ = masses[0]
    return measure_slope_J(fld, rmin, rmax, Nr=60, cx=cx, cy=cy, cz=cz)

# --------------------------- orbit (freeze field) ---------------------------

def plot_periodic_path(ax, X, Y, L):
    """2D seam-aware polyline (XY projection)."""
    X = np.asarray(X); Y = np.asarray(Y)
    if X.size < 2:
        ax.plot(X, Y, lw=1.2); return
    jump = (np.abs(np.diff(X)) > 0.5*L) | (np.abs(np.diff(Y)) > 0.5*L)
    s = 0
    for i,c in enumerate(jump,1):
        if c:
            ax.plot(X[s:i], Y[s:i], lw=1.2)
            s = i
    ax.plot(X[s:], Y[s:], lw=1.2)

def min_image(d, L):
    """Map displacement to the interval [-L/2, L/2)."""
    return (d + 0.5*L) % L - 0.5*L

def test_orbit_3d(N=128, L=64.0, masses=[(32,32,32,1.0)], eta=1.0, kappa=1.0, tau=0.2, c=1.0,
                  dt=0.02, warm=6000, steps=8000, r0=10.0, vtheta=0.32, sigma=1.0,
                  project_curl_free=False, orbit_from_potential=False):
    fld = CollapseField3D(N=N, L=L, tau=tau, c=c, eta=eta, kappa=kappa, sigma=sigma)
    fld.set_masses(masses)

    # evolve to steady field
    for _ in range(int(warm)):
        fld.step(dt)

    # Optional: project J for J-based readout
    if project_curl_free and not orbit_from_potential:
        project_curl_free_J(fld)
        print("[curl_free] J projected before orbit readout")

    # Build acceleration sampler
    if orbit_from_potential:
        # a = -∇(κ c^2 μ)  (conservative)
        gx, gy, gz = fld.spectral_grad_mu()
        aX = -kappa*(c**2)*gx
        aY = -kappa*(c**2)*gy
        aZ = -kappa*(c**2)*gz
        def accel_sample(xs, ys, zs):
            return (fld.sample_array(aX, xs,ys,zs),
                    fld.sample_array(aY, xs,ys,zs),
                    fld.sample_array(aZ, xs,ys,zs))
        acc_title = "a = -∇(κ c² μ)"
    else:
        # a = +κ J  (use projected J if requested)
        aX = kappa * fld.Jx
        aY = kappa * fld.Jy
        aZ = kappa * fld.Jz
        def accel_sample(xs, ys, zs):
            return (fld.sample_array(aX, xs,ys,zs),
                    fld.sample_array(aY, xs,ys,zs),
                    fld.sample_array(aZ, xs,ys,zs))
        acc_title = "a = + κ J"

    # energy from Φ_em = κ c² μ
    def phi_em(x):
        return kappa * c*c * fld.sample_mu(x[0], x[1], x[2])

    # initial state: position offset from center in XY plane, tangential v
    cx,cy,cz,_ = masses[0]
    x = np.array([ (cx + r0) % L, cy % L, cz % L ], dtype=float)
    v = np.array([ 0.0, vtheta, 0.0 ], dtype=float)

    xs = np.zeros((steps,3)); E = np.zeros(steps); Lnorm = np.zeros(steps)

    def leapfrog(x, v, dt):
        ax, ay, az = accel_sample(x[0], x[1], x[2])
        a  = np.array([ax, ay, az], dtype=float)
        v += 0.5*dt*a
        x  = (x + dt*v) % L
        ax2, ay2, az2 = accel_sample(x[0], x[1], x[2])
        a2 = np.array([ax2, ay2, az2], dtype=float)
        v += 0.5*dt*a2
        return x, v

    # plots
    fig, ax = plt.subplots(figsize=(5.4,5.0), dpi=140)
    plot_periodic_path(ax, xs[:,0], xs[:,1], L)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0,L); ax.set_ylim(0,L)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Emergent orbit (XY projection) under {acc_title}")
    plt.tight_layout(); plt.savefig("orbit_emergent_3d.png"); plt.close(fig)

    for i in range(steps):
        xs[i] = x
        E[i]  = 0.5*np.dot(v,v) + phi_em(x)
        
        if i == 0:
            print("[orbit] min_image seam-aware L enabled")

        # seam-aware r = x − center (minimum image)
        rx = min_image(x[0] - cx, L)
        ry = min_image(x[1] - cy, L)
        rz = min_image(x[2] - cz, L)
        rvec = np.array([rx, ry, rz], dtype=float)
        Lvec = np.cross(rvec, v)
        Lnorm[i] = np.linalg.norm(Lvec)

        x, v = leapfrog(x, v, dt)
    
    t = np.arange(steps)*dt
    dE = (E - E[0]) / max(1e-12, abs(E[0]))
    dL = (Lnorm - Lnorm[0]) / max(1e-12, abs(Lnorm[0]))
    plt.figure(figsize=(5.6,3.6), dpi=140); plt.plot(t, dE); plt.axhline(0, color='k', lw=0.8, alpha=0.5)
    plt.xlabel("time"); plt.ylabel("(E/E0)-1"); plt.title("Energy drift (emergent 3D)")
    plt.tight_layout(); plt.savefig("energy_emergent_3d.png"); plt.close()
    plt.figure(figsize=(5.6,3.6), dpi=140); plt.plot(t, dL); plt.axhline(0, color='k', lw=0.8, alpha=0.5)
    plt.xlabel("time"); plt.ylabel("(‖L‖/‖L‖0)-1"); plt.title("Angular momentum drift (emergent 3D)")
    plt.tight_layout(); plt.savefig("angmom_emergent_3d.png"); plt.close()

    print(f"[orbit_3d] mean |ΔE/E0| ≈ {np.mean(np.abs(dE)):.3e}, mean |Δ‖L‖/‖L‖0| ≈ {np.mean(np.abs(dL)):.3e}")

# --------------------------- CLI helpers ---------------------------

def parse_masses_3d(spec, L):
    """
    spec: "x,y,z,m; x,y,z,m; ..." positions in box coords.
    """
    items = []
    if not spec:
        return items
    for part in spec.split(";"):
        p = part.strip()
        if not p: continue
        toks = [t.strip() for t in p.split(",")]
        if len(toks) != 4:
            raise ValueError("Each mass must be x,y,z,m")
        x,y,z,m = map(float, toks)
        items.append((x%L, y%L, z%L, m))
    return items

# ======================= HYPERCHARGED PARALLEL SWEEP =======================

PIN_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

def _run_cmd(cmd, timeout=None, cwd=None):
    """Run a child command with pinned BLAS/FFT threads; capture tail of stdout/stderr."""
    env = os.environ.copy()
    env.update(PIN_ENV)
    t0 = time.time()
    p = None
    try:
        p = subprocess.run(
            shlex.split(cmd),
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "rc": p.returncode,
            "dt": time.time() - t0,
            "stdout": p.stdout[-20000:],
            "stderr": p.stderr[-20000:],
        }
    except subprocess.TimeoutExpired:
        return {
            "cmd": cmd,
            "rc": 124,
            "dt": time.time() - t0,
            "stdout": "" if p is None else (p.stdout[-20000:] if p.stdout else ""),
            "stderr": "TIMEOUT",
        }

def _parse_kv_list(s):
    """Parse 'key=value1,value2 ; key2=...' into dict of key -> list[str]."""
    d = {}
    if not s:
        return d
    for chunk in s.split(";"):
        c = chunk.strip()
        if not c:
            continue
        if "=" not in c:
            raise ValueError(f"Bad spec: '{c}' (expected key=value[,value])")
        k, v = c.split("=", 1)
        k = k.strip()
        vals = [x.strip() for x in v.split(",") if x.strip()]
        d[k] = vals
    return d

def _cartesian_dicts(grid_dict):
    """Yield dicts for the Cartesian product of grid_dict lists."""
    from itertools import product
    keys = list(grid_dict.keys())
    if not keys:
        yield {}
        return
    for combo in product(*[grid_dict[k] for k in keys]):
        yield {k: combo[i] for i, k in enumerate(keys)}

def _extract_slope(stdout_tail):
    slope = None
    for line in stdout_tail.splitlines():
        if "[slope_3d]" in line and "fitted slope" in line and "≈" in line:
            try:
                slope = float(line.split("≈", 1)[1].split()[0])
                break
            except Exception:
                pass
    return slope

def _extract_orbit_metrics(stdout_tail):
    edrift = None
    ldrift = None
    for line in stdout_tail.splitlines():
        if line.startswith("[orbit_3d]"):
            # If you later print specific tokens, parse them here.
            pass
    return edrift, ldrift

def _build_child_cmd(kind, args_dict):
    exe = sys.executable
    me  = Path(__file__).name

    masses = args_dict.get("masses", "32,32,32,1.0")
    if isinstance(masses, (list, tuple)):
        masses = ",".join(map(str, masses))

    base = (
        f'"{exe}" {me} --test {kind} '
        f'--N {args_dict.get("N", 160)} --L {args_dict.get("L", 80)} '
        f'--sigma {args_dict.get("sigma", 1.0)} --tau {args_dict.get("tau", 0.2)} '
        f'--dt {args_dict.get("dt", 0.05)} --warm {args_dict.get("warm", 7000)} '
        f'--masses "{masses}"'
    )

    if kind == "slope":
        base += f' --rmin {args_dict.get("rmin", 5)} --rmax {args_dict.get("rmax", 20)}'
    else:
        base += f' --steps {args_dict.get("steps", 10000)} --r0 {args_dict.get("r0", 10.0)} --vtheta {args_dict.get("vtheta", 0.32)}'

    # pass-through boolean flags if present / truthy
    if str(args_dict.get("project_curl_free", "")).lower() in ("1","true","yes","on"):
        base += " --project_curl_free"
    if kind == "orbit" and str(args_dict.get("orbit_from_potential", "")).lower() in ("1","true","yes","on"):
        base += " --orbit_from_potential"

    return base

def _job_runner(kind, job_cfg):
    cmd = _build_child_cmd(kind, job_cfg)
    res = _run_cmd(cmd, timeout=float(job_cfg.get("timeout_sec", 3600)))
    metrics = {}
    if kind == "slope":
        metrics["slope"] = _extract_slope(res["stdout"])
    else:
        ed, ld = _extract_orbit_metrics(res["stdout"])
        if ed is not None: metrics["energy_drift"] = ed
        if ld is not None: metrics["angmom_drift"] = ld
    res["metrics"] = metrics
    res["params"] = {k: v for k, v in job_cfg.items() if k != "timeout_sec"}
    return res

def run_sweep(kind, grid_spec, const_spec, out_dir="sweeps",
              max_workers=None, timeout_sec=3600.0,
              pass_eps=0.05, leaderboard_n=8, save_logs=True):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    grid   = _parse_kv_list(grid_spec)
    consts = {**_parse_kv_list(const_spec)}
    # Flatten singletons in consts
    for k, vals in list(consts.items()):
        if isinstance(vals, list) and len(vals) == 1:
            consts[k] = vals[0]
    consts["timeout_sec"] = timeout_sec

    jobs = [{**g, **consts} for g in _cartesian_dicts(grid)]
    if not jobs:
        print("[sweep] No jobs in grid — nothing to run.")
        return None

    workers = max_workers or max(1, os.cpu_count() // 2)
    print(f"[sweep] kind={kind} jobs={len(jobs)} workers={workers} out_dir={out_dir}")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut2cfg = {ex.submit(_job_runner, kind, j): j for j in jobs}
        for fut in as_completed(fut2cfg):
            r = fut.result()
            results.append(r)
            ok = (r["rc"] == 0)
            badge = "✅" if ok else "⚠️ "
            extra = ""
            if kind == "slope" and r["metrics"].get("slope") is not None:
                extra = f"  slope={r['metrics']['slope']:+.6f}"
            print(f"{badge} {r['cmd']}{extra}")

    wall = time.time() - t0
    total = len(results)
    ok_count = sum(1 for r in results if r["rc"] == 0)
    throughput = total / wall if wall > 0 else 0.0

    slopes = [r.get("metrics", {}).get("slope") for r in results if r.get("metrics", {}).get("slope") is not None]
    mean_slope = (sum(slopes)/len(slopes)) if slopes else None

    pass_count = None
    if kind == "slope" and slopes:
        pass_count = sum(1 for r in results if abs(r["metrics"]["slope"] + 2.0) <= pass_eps)

    report = {
        "summary": {
            "kind": kind,
            "jobs": total,
            "ok": ok_count,
            "pass_eps": pass_eps if kind == "slope" else None,
            "pass": pass_count if kind == "slope" else None,
            "wall_sec": wall,
            "throughput_jobs_per_sec": throughput,
            "mean_slope": mean_slope if kind == "slope" else None,
        },
        "grid": grid,
        "const": consts,
        "results": results,
    }

    outp = Path(out_dir) / f"{kind}_sweep_{int(time.time())}.json"
    with open(outp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[sweep] saved {outp}")

    # Leaderboard (slope)
    if kind == "slope" and slopes:
        ranked = []
        for r in results:
            s = r.get("metrics", {}).get("slope")
            if s is None: continue
            ranked.append((abs(s + 2.0), s, r["cmd"]))
        ranked.sort(key=lambda t: t[0])
        top = ranked[:leaderboard_n]
        print("\n=== Top slope fits (closest to -2) ===")
        for d, s, cmd in top:
            print(f"  slope={s:+.6f}  Δ={d:.6f}   {cmd}")

    return str(outp)

# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["slope","orbit"], help="Run a single test")
    ap.add_argument("--N", type=int, default=160)
    ap.add_argument("--L", type=float, default=80.0)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--c",   type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--masses", type=str, default="32,32,32,1.0")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--warm", type=int, default=7000)
    ap.add_argument("--rmin", type=float, default=5.0)
    ap.add_argument("--rmax", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--r0", type=float, default=10.0)
    ap.add_argument("--vtheta", type=float, default=0.32)
    # NEW flags
    ap.add_argument("--project_curl_free", action="store_true",
                    help="One-shot Helmholtz projection of J before readouts (slope/orbit).")
    ap.add_argument("--orbit_from_potential", action="store_true",
                    help="Use a = -∇(κ c^2 μ) (conservative) instead of a = +κ J for orbits.")

    # Sweep mode
    ap.add_argument("--sweep", choices=["slope","orbit"], help="Run a parallel parameter sweep")
    ap.add_argument("--grid", type=str, help='Grid spec: e.g. "N=160,192; L=80; sigma=0.6,1.0; tau=0.15,0.20"')
    ap.add_argument("--const", type=str, default="", help='Constants: e.g. "dt=0.05; warm=7000; rmin=5; rmax=20"')
    ap.add_argument("--out_dir", type=str, default="sweeps")
    ap.add_argument("--max_workers", type=int, default=None)
    ap.add_argument("--timeout_sec", type=float, default=3600.0)
    ap.add_argument("--pass_eps", type=float, default=0.05, help="|slope + 2| ≤ pass_eps marks PASS")
    ap.add_argument("--leaderboard_n", type=int, default=8)

    args = ap.parse_args()

    # Sweep branch
    if args.sweep:
        if not args.grid:
            raise SystemExit("ERROR: --sweep requires --grid")
        run_sweep(
            kind=args.sweep,
            grid_spec=args.grid,
            const_spec=args.const,
            out_dir=args.out_dir,
            max_workers=args.max_workers,
            timeout_sec=args.timeout_sec,
            pass_eps=args.pass_eps,
            leaderboard_n=args.leaderboard_n,
        )
        return 0

    # Single-run branch
    masses = parse_masses_3d(args.masses, args.L) if "parse_masses_3d" in globals() else args.masses
    if args.test == "slope":
        p = test_slope_3d(
            N=args.N, L=args.L, masses=masses,
            eta=args.eta, kappa=args.kappa, tau=args.tau, c=args.c,
            dt=args.dt, warm=args.warm, rmin=args.rmin, rmax=args.rmax,
            sigma=args.sigma, project_curl_free=args.project_curl_free
        )
        status = "PASS" if abs(p + 2.0) <= args.pass_eps else "WARN"
        print(f"[result] {status} slope={p:+.6f} (target -2)")
    elif args.test == "orbit":
        test_orbit_3d(
            N=args.N, L=args.L, masses=masses,
            eta=args.eta, kappa=args.kappa, tau=args.tau, c=args.c,
            dt=args.dt, warm=args.warm, steps=args.steps,
            r0=args.r0, vtheta=args.vtheta, sigma=args.sigma,
            project_curl_free=args.project_curl_free,
            orbit_from_potential=args.orbit_from_potential
        )
    else:
        ap.print_help()
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
