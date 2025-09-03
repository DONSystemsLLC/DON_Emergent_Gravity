#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
don_orbit_live_lambda.py — Λ evolution + multi-particle orbits under Φ(x,y,t)

Modes
-----
live         : evolve Λ via PDE (stepper: pde_fft | pde_fd) while pushing many particles
live_relax   : evolve Φ toward Poisson Φ_ref by exponential relaxation (no ∇²)
frozen       : steady Poisson Φ (no field stepping during orbit loop)

Steppers
--------
pde_fft  : spectral semi-implicit (robust; recommended)
pde_fd   : finite-difference semi-implicit (kept for reference)
relax    : same as --mode live_relax (per-step relaxation toward Φ_ref)

Field safety
------------
- Spectral semi-implicit for stiff -Ψ/τ decay (pde_fft)
- Pre/post cleaning & clipping of Φ, Ψ, ∇Φ (envelope tied to std(Φ_ref))
- Field substeps per particle step (--field_substeps)
- Conservative dt: dt_field = min(CFL·dx/cL, 0.25·τ)

Density controls
----------------
Primary blob: --rho_cx --rho_cy --rho_soft   (fractions of box; default 0.35, 0.50, 0.06)
Second blob : --rho_m2 (mass scale; 0 disables), --rho_cx2 --rho_cy2 --rho_soft2
Global scale: --rho_amp

Particles
---------
- Vectorized leapfrog (N tracers; non self-interacting).
- Initial layouts: ring | gaussian | random | grid.
- Trajectory storage stride (--store_every) and plotting subset (--n_plot).

Frames / movie
--------------
--save_frames (PNG frames of Φ + tracers), --make_mp4 (requires ffmpeg)

Outputs
-------
- phi_convergence.png  (Φ residual vs Poisson during warmup when applicable)
- particles_paths.png  (subset of trajectories; seam-aware)
- particle_single.png  (first plotted tracer)
- particles_final.png  (final positions, colored by speed)
- energy_<mode>.png    (mean tracer energy vs time)
- frames/<...>.png     (if --save_frames), and mp4 (if --make_mp4)
"""

import argparse
import os, shutil, subprocess
import numpy as np
import matplotlib.pyplot as plt

# --- robust ffmpeg discovery (system or pip imageio-ffmpeg) ---
try:
    import imageio_ffmpeg as _ioffmpeg
    _HAS_IMAGEIO_FFMPEG = True
except Exception:
    _HAS_IMAGEIO_FFMPEG = False

def _get_ffmpeg_exe():
    """Return a working ffmpeg path or None."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    if _HAS_IMAGEIO_FFMPEG:
        try:
            return _ioffmpeg.get_ffmpeg_exe()  # downloads a static ffmpeg if needed
        except Exception:
            return None
    return None

# ---------- FD ops (used only by pde_fd) ----------

def laplacian_periodic(phi, dx, dy):
    return (
        (np.roll(phi, +1, 0) - 2*phi + np.roll(phi, -1, 0))/dx**2 +
        (np.roll(phi, +1, 1) - 2*phi + np.roll(phi, -1, 1))/dy**2
    )

def grad_periodic(phi, dx, dy):
    gx = (np.roll(phi, -1, 0) - np.roll(phi, +1, 0)) / (2*dx)
    gy = (np.roll(phi, -1, 1) - np.roll(phi, +1, 1)) / (2*dy)
    return gx, gy

# ---------- spectral helpers ----------

def spectral_gradients(phi, Lx, Ly):
    """Smooth periodic ∇φ via FFT (exact on the grid)."""
    Nx, Ny = phi.shape
    kx = np.fft.fftfreq(Nx, d=Lx/Nx) * 2*np.pi
    ky = np.fft.fftfreq(Ny, d=Ly/Ny) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    phi_k = np.fft.fft2(phi)
    gx = np.fft.ifft2(1j*KX * phi_k).real
    gy = np.fft.ifft2(1j*KY * phi_k).real
    return gx, gy

def spectral_phi_sample(phi, Lx, Ly, xs, ys, up=4):
    """
    Periodic spectral interpolation of φ at many points xs,ys:
    - zero-pad FFT by 'up' factor
    - inverse FFT to a fine grid
    - bilinear on that fine grid
    """
    Nx, Ny = phi.shape
    Fx = np.fft.fft2(phi)

    # zero-pad in FFT (centered, 2D)
    pad = np.zeros((up*Nx, up*Ny), dtype=complex)
    kx_half, ky_half = Nx//2, Ny//2
    pad[:kx_half, :ky_half] = Fx[:kx_half, :ky_half]
    pad[:kx_half, -ky_half:] = Fx[:kx_half, -ky_half:]
    pad[-kx_half:, :ky_half] = Fx[-kx_half:, :ky_half]
    pad[-kx_half:, -ky_half:] = Fx[-kx_half:, -ky_half:]

    phi_up = np.fft.ifft2(pad).real * (up*up)

    Nx_u, Ny_u = phi_up.shape
    dxu, dyu = Lx/Nx_u, Ly/Ny_u

    xs = np.asarray(xs); ys = np.asarray(ys)
    fx = (xs/dxu) % Nx_u; fy = (ys/dyu) % Ny_u
    i0 = np.floor(fx).astype(int) % Nx_u; j0 = np.floor(fy).astype(int) % Ny_u
    i1 = (i0+1) % Nx_u;                 j1 = (j0+1) % Ny_u
    tx = fx - np.floor(fx);             ty = fy - np.floor(fy)

    p00 = phi_up[i0,j0]; p10 = phi_up[i1,j0]; p01 = phi_up[i0,j1]; p11 = phi_up[i1,j1]
    val = (1-ty)*((1-tx)*p00 + tx*p10) + ty*((1-tx)*p01 + tx*p11)
    return np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)

# ---------- Poisson (spectral) ----------

def poisson_fft_2d(rho, Lx, Ly, G=1.0):
    Nx, Ny = rho.shape
    kx = np.fft.fftfreq(Nx, d=Lx/Nx) * 2*np.pi
    ky = np.fft.fftfreq(Ny, d=Ly/Ny) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    rho_k = np.fft.fft2(rho)
    with np.errstate(divide='ignore', invalid='ignore'):
        phi_k = np.where(K2>0.0, 4*np.pi*G * rho_k / K2, 0.0)
    return np.fft.ifft2(phi_k).real

# ---------- Λ-field (spectral stepper) ----------

class LambdaField2D:
    def __init__(self, Nx=256, Ny=256, Lx=1.0, Ly=1.0, tau=0.05, cL=1.0, G=1.0,
                 clip_sigma=8.0, phi_scale=1.0):
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx/Nx, Ly/Ny
        self.tau = float(tau)
        self.cL  = float(cL)
        self.G   = float(G)
        self.clip_sigma = float(clip_sigma)
        self.phi_scale  = float(max(1.0, phi_scale))
        self.rho = np.zeros((Nx,Ny), np.float64)
        self.phi = np.zeros((Nx,Ny), np.float64)   # Φ
        self.psi = np.zeros((Nx,Ny), np.float64)   # Ψ = ∂t Φ
        # spectral grids
        kx = np.fft.fftfreq(Nx, d=Lx/Nx) * 2*np.pi
        ky = np.fft.fftfreq(Ny, d=Ly/Ny) * 2*np.pi
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2

    def set_density(self, rho):
        assert rho.shape == (self.Nx, self.Ny)
        self.rho = rho.astype(np.float64, copy=True)

    def set_phi_scale_from(self, phi_ref):
        self.phi_scale = float(max(1.0, np.std(phi_ref)))

    def clean_clip(self):
        self.phi = np.nan_to_num(self.phi, nan=0.0, posinf=0.0, neginf=0.0)
        self.psi = np.nan_to_num(self.psi, nan=0.0, posinf=0.0, neginf=0.0)
        bound = self.clip_sigma * self.phi_scale
        np.clip(self.phi, -bound, +bound, out=self.phi)
        np.clip(self.psi, -2*bound, +2*bound, out=self.psi)

    def step_pde_fd(self, dt):
        """Finite-difference semi-implicit Ψ, explicit Φ (kept for reference)."""
        self.clean_clip()
        lap = laplacian_periodic(self.phi, self.dx, self.dy)
        A = self.cL**2 * lap - 4.0*np.pi*self.G*self.rho
        fac = 1.0 / (1.0 + dt/self.tau)
        self.psi = fac * (self.psi + (dt/self.tau) * A)
        self.phi = self.phi + dt * self.psi
        self.clean_clip()

    def step_pde_fft(self, dt):
        """
        Spectral semi-implicit:
           lap(Φ_n) = F^-1[-k^2 Φ_k]
           RHS = Ψ_n + (dt/τ)*(cL^2 lap(Φ_n) - 4πGρ)
           Ψ_{n+1} = RHS / (1 + dt/τ)
           Φ_{n+1} = Φ_n + dt Ψ_{n+1}
        """
        self.clean_clip()
        phi_k = np.fft.fft2(self.phi)
        lap_k = -self.K2 * phi_k
        lap = np.fft.ifft2(lap_k).real
        RHS = self.psi + (dt/self.tau) * (self.cL**2 * lap - 4.0*np.pi*self.G*self.rho)
        self.psi = RHS / (1.0 + dt/self.tau)
        self.phi = self.phi + dt * self.psi
        self.clean_clip()

    def step_relax_to_ref(self, dt, phi_ref, relax_rate=0.2):
        """Safe relaxation toward Φ_ref: Φ←(1-β)Φ+βΦ_ref, β=clamp(relax_rate·dt, 0..1)."""
        beta = float(np.clip(relax_rate * dt, 0.0, 1.0))
        phi_old = self.phi
        self.phi = (1.0 - beta) * self.phi + beta * phi_ref
        self.psi = (self.phi - phi_old) / max(dt, 1e-12)
        self.clean_clip()

    def accel(self):
        gx, gy = spectral_gradients(self.phi, self.Lx, self.Ly)  # smoother
        return -gx, -gy

    def poisson_reference(self):
        return poisson_fft_2d(self.rho, self.Lx, self.Ly, self.G)

    def stable_dt(self, cfl=0.45):
        return min(cfl*min(self.dx, self.dy)/max(self.cL,1e-12), 0.25*self.tau)

# ---------- density, sampling & plotting helpers ----------

def build_density(
    Nx, Ny, Lx, Ly,
    m1=1.0, cx1=0.35, cy1=0.50, soft1=0.06,
    m2=0.0, cx2=0.70, cy2=0.55, soft2=0.06,
    amp=0.5
):
    """Two Plummer-like blobs at (cx,cy) in fractional box coords; mean-removed, scaled by amp."""
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    Cx1, Cy1 = cx1*Lx, cy1*Ly
    a1 = soft1*Lx
    r1 = np.sqrt((X-Cx1)**2 + (Y-Cy1)**2)
    rho = m1 * (1 + (r1/a1)**2)**(-2.5)

    if m2 != 0.0:
        Cx2, Cy2 = cx2*Lx, cy2*Ly
        a2 = soft2*Lx
        r2 = np.sqrt((X-Cx2)**2 + (Y-Cy2)**2)
        rho += m2 * (1 + (r2/a2)**2)**(-2.5)

    rho -= rho.mean()
    return amp * rho

def sample_many(ax, ay, Lx, Ly, xs, ys):
    """Vectorized bilinear sampling for many positions."""
    Nx, Ny = ax.shape
    dx, dy = Lx/Nx, Ly/Ny
    xs = np.asarray(xs); ys = np.asarray(ys)
    fx = (xs/dx) % Nx; fy = (ys/dy) % Ny
    i0 = np.floor(fx).astype(int) % Nx
    j0 = np.floor(fy).astype(int) % Ny
    i1 = (i0+1) % Nx
    j1 = (j0+1) % Ny
    tx = fx - np.floor(fx); ty = fy - np.floor(fy)

    ax00 = ax[i0, j0]; ax10 = ax[i1, j0]; ax01 = ax[i0, j1]; ax11 = ax[i1, j1]
    ay00 = ay[i0, j0]; ay10 = ay[i1, j0]; ay01 = ay[i0, j1]; ay11 = ay[i1, j1]

    axp = (1-ty)*((1-tx)*ax00 + tx*ax10) + ty*((1-tx)*ax01 + tx*ax11)
    ayp = (1-ty)*((1-tx)*ay00 + tx*ay10) + ty*((1-tx)*ay01 + tx*ay11)

    axp = np.nan_to_num(axp, nan=0.0, posinf=0.0, neginf=0.0)
    ayp = np.nan_to_num(ayp, nan=0.0, posinf=0.0, neginf=0.0)
    return axp.astype(float), ayp.astype(float)

def plot_periodic_path(ax, xx, yy, Lx, Ly, **kw):
    """Draw a 2D path but break the polyline wherever it crosses a periodic seam."""
    xx = np.asarray(xx); yy = np.asarray(yy)
    if xx.size < 2:
        ax.plot(xx, yy, **kw); return
    jmp = (np.abs(np.diff(xx)) > 0.5*Lx) | (np.abs(np.diff(yy)) > 0.5*Ly)
    start = 0
    for i, cut in enumerate(jmp, 1):
        if cut:
            ax.plot(xx[start:i], yy[start:i], **kw)
            start = i
    ax.plot(xx[start:], yy[start:], **kw)

# ---------- frame writer ----------

class FrameWriter:
    def __init__(self, frame_dir, Lx, Ly, vmin, vmax, n_plot=400):
        self.dir = frame_dir
        self.Lx, self.Ly = Lx, Ly
        self.vmin, self.vmax = vmin, vmax
        self.n_plot = n_plot
        os.makedirs(self.dir, exist_ok=True)
        self.idx = 0

    def save(self, phi, x, y, tag=""):
        fig, ax = plt.subplots(figsize=(6.0,5.2), dpi=140)
        im = ax.imshow(phi.T, origin="lower", extent=[0,self.Lx,0,self.Ly],
                       cmap="viridis", vmin=self.vmin, vmax=self.vmax, interpolation="nearest")
        # decimate particle set for frame
        if x.size > self.n_plot:
            step = max(1, x.size // self.n_plot)
            xs = x[::step]; ys = y[::step]
        else:
            xs, ys = x, y
        ax.scatter(xs, ys, s=6, c="w", edgecolors="none", alpha=0.85)
        ax.set_xlim(0,self.Lx); ax.set_ylim(0,self.Ly)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"Φ + tracers {tag}")
        cbar = fig.colorbar(im, ax=ax, shrink=0.88); cbar.set_label("Φ")
        fig.tight_layout()
        out = os.path.join(self.dir, f"frame_{self.idx:05d}.png")
        fig.savefig(out); plt.close(fig)
        self.idx += 1

def build_mp4(frame_dir, out_mp4="lambda_orbit.mp4", fps=24):
    exe = _get_ffmpeg_exe()
    if not exe:
        print("[frames] ffmpeg not available (system or imageio-ffmpeg). Skipping mp4.")
        return
    cmd = [
        exe, "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_mp4,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[frames] wrote {out_mp4}")
    except subprocess.CalledProcessError as e:
        print(f"[frames] ffmpeg failed: {e}")

# ---------- particle initializations (shape-safe grid) ----------

def init_particles(n, mode, Lx, Ly, rng, r0=0.15, sigma=0.05):
    if mode == "ring":
        theta = rng.uniform(0, 2*np.pi, n)
        r = r0 * np.ones(n)
        x = 0.55*Lx + r*np.cos(theta)
        y = 0.50*Ly + r*np.sin(theta)
        vmag = 0.25
        vx = -vmag*np.sin(theta); vy = vmag*np.cos(theta)

    elif mode == "gaussian":
        x = 0.55*Lx + rng.normal(0, sigma*Lx, n)
        y = 0.50*Ly + rng.normal(0, sigma*Ly, n)
        vx = rng.normal(0, 0.1, n); vy = rng.normal(0, 0.1, n)

    elif mode == "grid":
        # ceil so m*m >= n, then cut to exactly n
        m = int(np.ceil(np.sqrt(n)))
        gx = np.linspace(0.4*Lx, 0.7*Lx, m, endpoint=True)
        gy = np.linspace(0.35*Ly, 0.65*Ly, m, endpoint=True)
        X, Y = np.meshgrid(gx, gy, indexing='ij')
        x = X.ravel()[:n]; y = Y.ravel()[:n]
        vx = np.zeros_like(x); vy = np.zeros_like(y)

    else:  # random
        x = rng.uniform(0, Lx, n); y = rng.uniform(0, Ly, n)
        vx = rng.normal(0, 0.15, n); vy = rng.normal(0, 0.15, n)

    return x, y, vx, vy

# ---------- main driver ----------

def run(args):
    Nx, Ny = args.Nx, args.Ny
    Lx, Ly = args.Lx, args.Ly
    rng = np.random.default_rng(args.seed)

    # Field & density
    rho = build_density(
        Nx, Ny, Lx, Ly,
        m1=1.0, cx1=args.rho_cx, cy1=args.rho_cy, soft1=args.rho_soft,
        m2=args.rho_m2, cx2=args.rho_cx2, cy2=args.rho_cy2, soft2=args.rho_soft2,
        amp=args.rho_amp
    )
    fld = LambdaField2D(Nx, Ny, Lx, Ly, tau=args.tau, cL=args.cL, G=args.G,
                        clip_sigma=args.clip_sigma, phi_scale=1.0)
    fld.set_density(rho)

    # Poisson reference and scale for clipping
    phi_ref = fld.poisson_reference()
    fld.set_phi_scale_from(phi_ref)

    if args.init_poisson or args.mode in ("frozen","live_relax"):
        fld.phi = phi_ref.copy(); fld.psi.fill(0.0); fld.clean_clip()

    dt_fld = fld.stable_dt(cfl=args.cfl)

    # Frames: setup writer & initial frame
    vmin = float(np.percentile(phi_ref, 2.0))
    vmax = float(np.percentile(phi_ref, 98.0))
    fw = None
    if args.save_frames:
        fw = FrameWriter(args.frame_dir, Lx, Ly, vmin, vmax, n_plot=args.frame_n_plot)
        fw.save(fld.phi if (args.init_poisson or args.mode!="frozen") else phi_ref, 
                np.array([]), np.array([]), tag="(init)")

    # Warmup to near-Poisson (only if live + stepper not 'relax' and not init_poisson)
    do_warm = (args.mode=="live" and args.stepper!="relax" and not args.init_poisson and args.warm_time>0.0)
    steps_warm = int(max(0.0, args.warm_time)/dt_fld) if do_warm else 0
    res_hist=[]
    for n in range(steps_warm):
        if args.stepper == "pde_fft":
            fld.step_pde_fft(dt_fld)
        else:
            fld.step_pde_fd(dt_fld)
        if (n % max(1, steps_warm//200) == 0) or (n == steps_warm-1):
            num = np.abs(fld.phi - phi_ref)
            den = np.maximum(np.abs(phi_ref), 1e-8)
            val = float((num/den).mean())
            res_hist.append(max(val, 1e-16))
        if fw and (n % max(1, args.frame_every//4) == 0):
            fw.save(fld.phi, np.array([]), np.array([]), tag=f"(warm {n})")

    if res_hist:
        t_axis = np.arange(len(res_hist)) * (steps_warm/max(1,len(res_hist))) * dt_fld
        plt.figure(figsize=(5.0,3.6), dpi=140)
        plt.semilogy(t_axis, res_hist, lw=1.5)
        plt.xlabel("time"); plt.ylabel("mean relative residual")
        plt.title("Φ convergence to Poisson (warmup)")
        plt.tight_layout(); plt.savefig("phi_convergence.png"); plt.close()

    # initial accel (spectral) & clip
    ax, ay = fld.accel()
    ax = np.nan_to_num(ax, nan=0.0, posinf=0.0, neginf=0.0)
    ay = np.nan_to_num(ay, nan=0.0, posinf=0.0, neginf=0.0)
    a_env = max(1.0, 10.0*np.std(ax) + 10.0*np.std(ay))
    np.clip(ax, -a_env, +a_env, out=ax); np.clip(ay, -a_env, +a_env, out=ay)

    # Build particles (shape-safe)
    N_req = args.n_particles
    x, y, vx, vy = init_particles(N_req, args.particles_init, Lx, Ly, rng,
                                  r0=args.ring_radius, sigma=args.gaussian_sigma)
    N = int(np.asarray(x).size)
    vx = np.asarray(vx).reshape(N)
    vy = np.asarray(vy).reshape(N)

    # Trajectory storage (subset & stride)
    stride = max(1, args.store_every)
    kkeep  = max(1, N // max(1, args.n_plot))
    keep_idx = np.arange(0, N, kkeep)[:args.n_plot]
    Tsteps = (args.steps_orbit + stride - 1)//stride
    paths = np.zeros((len(keep_idx), Tsteps, 2), float)
    store_t = 0

    # Mean energy trace
    meanE = np.zeros(args.steps_orbit, float)

    # Orbit integration: sub-cycle field, then vectorized leapfrog on tracers
    dtp   = dt_fld
    sub   = max(1, args.field_substeps)

    mode_title = {"live":"LIVE Λ-field","live_relax":"LIVE (relax→Poisson)","frozen":"FROZEN Poisson Φ"}
    tag = {"live":"live","live_relax":"relax","frozen":"frozen"}[args.mode]

    if fw and args.mode=="frozen":
        fw.save(fld.phi, x, y, tag="(start)")

    for n in range(args.steps_orbit):
        if args.mode == "live":
            for _ in range(sub):
                if args.stepper == "pde_fft":
                    fld.step_pde_fft(dtp/sub)
                elif args.stepper == "pde_fd":
                    fld.step_pde_fd(dtp/sub)
                else:
                    fld.step_relax_to_ref(dtp/sub, phi_ref, relax_rate=args.relax_rate)
            ax, ay = fld.accel()
            ax = np.nan_to_num(ax, nan=0.0, posinf=0.0, neginf=0.0)
            ay = np.nan_to_num(ay, nan=0.0, posinf=0.0, neginf=0.0)
            a_env = max(1.0, 10.0*np.std(ax) + 10.0*np.std(ay))
            np.clip(ax, -a_env, +a_env, out=ax); np.clip(ay, -a_env, +a_env, out=ay)

        elif args.mode == "live_relax":
            for _ in range(sub):
                fld.step_relax_to_ref(dtp/sub, phi_ref, relax_rate=args.relax_rate)
            ax, ay = fld.accel()
            ax = np.nan_to_num(ax, nan=0.0, posinf=0.0, neginf=0.0)
            ay = np.nan_to_num(ay, nan=0.0, posinf=0.0, neginf=0.0)
            a_env = max(1.0, 10.0*np.std(ax) + 10.0*np.std(ay))
            np.clip(ax, -a_env, +a_env, out=ax); np.clip(ay, -a_env, +a_env, out=ay)

        # Half-kick
        axp, ayp = sample_many(ax, ay, Lx, Ly, x, y)
        vx += 0.5*dtp*axp; vy += 0.5*dtp*ayp

        # Drift (periodic)
        x  = (x + dtp*vx) % Lx
        y  = (y + dtp*vy) % Ly

        # Full-kick
        axp, ayp = sample_many(ax, ay, Lx, Ly, x, y)
        vx += 0.5*dtp*axp; vy += 0.5*dtp*ayp

        # Energy (mean; spectral φ sampling for smoothness)
        phi_part = spectral_phi_sample(fld.phi, Lx, Ly, x, y, up=4)
        e_now = 0.5*(vx*vx + vy*vy) + phi_part
        meanE[n] = float(np.mean(e_now))

        # Store subset paths
        if (n % stride) == 0:
            paths[:, store_t, 0] = x[keep_idx]
            paths[:, store_t, 1] = y[keep_idx]
            store_t += 1

        # Frame save
        if fw and (n % args.frame_every == 0):
            fw.save(fld.phi, x, y, tag=f"(step {n})")

    # ---- Plots ----
    # 1) Paths (subset; seam-aware)
    fig, axp = plt.subplots(figsize=(6.2,5.6), dpi=140)
    for i in range(paths.shape[0]):
        plot_periodic_path(axp, paths[i,:store_t,0], paths[i,:store_t,1], Lx, Ly, lw=1.0, alpha=0.9)
    axp.set_aspect('equal', adjustable='box')
    axp.set_xlim(0,Lx); axp.set_ylim(0,Ly)
    axp.set_xlabel("x"); axp.set_ylabel("y")
    axp.set_title("Tracer trajectories (subset)")
    fig.tight_layout(); fig.savefig("particles_paths.png"); plt.close(fig)

    # 2) Single tracer (first kept), seam-aware
    fig, ax1 = plt.subplots(figsize=(5.2,4.8), dpi=140)
    plot_periodic_path(ax1, paths[0,:store_t,0], paths[0,:store_t,1], Lx, Ly, lw=1.2)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim(0,Lx); ax1.set_ylim(0,Ly)
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.set_title(f"Particle orbit ({mode_title[args.mode]})")
    fig.tight_layout(); fig.savefig("particle_single.png"); plt.close(fig)

    # 3) Final positions colored by speed
    speed = np.sqrt(vx*vx + vy*vy)
    plt.figure(figsize=(6.2,5.6), dpi=140)
    plt.scatter(x, y, c=speed, s=6, cmap="viridis", edgecolors="none")
    plt.colorbar(label="speed")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0,Lx); plt.ylim(0,Ly)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Final particle positions (colored by speed)")
    plt.tight_layout(); plt.savefig("particles_final.png"); plt.close()

    # 4) Mean energy trace
    t_axis = np.arange(args.steps_orbit) * dt_fld
    denE = max(1e-8, abs(meanE[0])) if np.isfinite(meanE[0]) else 1.0
    plt.figure(figsize=(5.6,3.6), dpi=140)
    plt.plot(t_axis, (meanE-meanE[0])/denE, lw=1.0)
    plt.xlabel("time"); plt.ylabel("⟨(E/E0)-1⟩")
    plt.title(f"Mean energy (mode={args.mode}, stepper={args.stepper})")
    plt.tight_layout(); plt.savefig(f"energy_{args.mode}.png"); plt.close()

    print("[nbody] wrote particle_single.png, particles_paths.png, particles_final.png, energy_*.png")
    if len(res_hist)>0: print("[nbody] wrote phi_convergence.png")
    if fw:
        print(f"[frames] wrote {fw.idx} PNG frames to {args.frame_dir}")
        if args.make_mp4:
            build_mp4(args.frame_dir, out_mp4=args.mp4_name, fps=args.mp4_fps)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live","live_relax","frozen"], default="live")
    ap.add_argument("--stepper", choices=["pde_fft","pde_fd","relax"], default="pde_fft",
                    help="Field stepper in LIVE mode (pde_fft recommended).")
    ap.add_argument("--Nx", type=int, default=256)
    ap.add_argument("--Ny", type=int, default=256)
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--Ly", type=float, default=1.0)
    ap.add_argument("--G",  type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--cL",  type=float, default=1.0)
    ap.add_argument("--cfl", type=float, default=0.35)
    ap.add_argument("--warm_time", type=float, default=0.6,
                    help="Warmup time before orbit if LIVE and not init_poisson.")
    ap.add_argument("--init_poisson", action="store_true",
                    help="Initialize Φ=Φ_ref, ψ=0.")
    # Density controls
    ap.add_argument("--rho_amp",   type=float, default=0.3,  help="Global density amplitude.")
    ap.add_argument("--rho_cx",    type=float, default=0.35, help="Primary blob center x (fraction of box).")
    ap.add_argument("--rho_cy",    type=float, default=0.50, help="Primary blob center y (fraction of box).")
    ap.add_argument("--rho_soft",  type=float, default=0.06, help="Primary blob softening (fraction of Lx).")
    ap.add_argument("--rho_m2",    type=float, default=0.0,  help="Second blob mass (0 disables).")
    ap.add_argument("--rho_cx2",   type=float, default=0.70, help="Second blob center x (fraction of box).")
    ap.add_argument("--rho_cy2",   type=float, default=0.55, help="Second blob center y (fraction of box).")
    ap.add_argument("--rho_soft2", type=float, default=0.06, help="Second blob softening (fraction of Lx).")
    ap.add_argument("--clip_sigma", type=float, default=8.0,
                    help="Φ/Ψ clipping envelope = clip_sigma * std(Φ_ref).")
    # Particles
    ap.add_argument("--n_particles", type=int, default=600, help="Number of tracers.")
    ap.add_argument("--particles_init", choices=["ring","gaussian","random","grid"], default="ring")
    ap.add_argument("--ring_radius", type=float, default=0.15, help="Ring radius (box units).")
    ap.add_argument("--gaussian_sigma", type=float, default=0.05, help="Gaussian pos std (fraction of box).")
    ap.add_argument("--store_every", type=int, default=20, help="Trajectory storage stride.")
    ap.add_argument("--n_plot", type=int, default=60, help="Number of trajectories to plot.")
    ap.add_argument("--field_substeps", type=int, default=6, help="Field substeps per particle step.")
    ap.add_argument("--relax_rate", type=float, default=0.2, help="LIVE_RELAX rate toward Φ_ref.")
    ap.add_argument("--steps_orbit", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    # Frames / movie
    ap.add_argument("--save_frames", action="store_true",
                    help="Save PNG frames of Φ + tracers during the run.")
    ap.add_argument("--frame_every", type=int, default=50,
                    help="Save a frame every N particle steps.")
    ap.add_argument("--frame_dir", type=str, default="frames",
                    help="Directory to write frames.")
    ap.add_argument("--frame_n_plot", type=int, default=400,
                    help="Max particles to plot per frame (decimated).")
    ap.add_argument("--make_mp4", action="store_true",
                    help="If set, attempt to build an mp4 from frames using ffmpeg.")
    ap.add_argument("--mp4_name", type=str, default="lambda_orbit.mp4",
                    help="Output mp4 file name (when --make_mp4).")
    ap.add_argument("--mp4_fps", type=int, default=24,
                    help="Frames per second for mp4.")
    args = ap.parse_args()
    run(args)
