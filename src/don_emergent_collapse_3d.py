 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
don_emergent_collapse_3d.py

3D DON collapse kernel:

  ∂t μ + ∇·J = S(x,t)
  τ ∂t J + J = - c^2 ∇μ

No Poisson, no 1/r^2 coded anywhere.

Conservative readout (for orbits):
  Φ_em = - κ c^2 μ   ⇒   a = -∇Φ_em = + κ c^2 ∇μ   (attractive, inward)
(J-readout also supported: a = + κ J)
"""

import argparse, json, time, shlex, subprocess, hashlib, re
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# JAX imports for optimization
try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np  # Fallback to numpy
    jit = lambda f: f  # No-op decorator

# --------------------------- 3D collapse field ---------------------------

class CollapseField3D:
    """
    Periodic box [0, L]^3 on a uniform cubic grid (N^3).

    State:
      μ(x,y,z,t)  -- memory density
      J(x,y,z,t)  -- memory flux = (Jx, Jy, Jz)

    Update scheme (semi-implicit in J-decay):
      J_{n+1} = (J_n + (dt/τ)*(-c^2 ∇μ_n)) / (1 + dt/τ)
      μ_{n+1} = μ_n + dt * ( -∇·J_{n+1} + S_n )

    Notes:
      • Conservative readout uses φ_em = - κ c^2 μ  ⇒  a = -∇φ_em = + κ c^2 ∇μ (inward).
      • Trilinear sampling is seam-aware (periodic) and numerically matched for φ and ∇φ.
    """

    def __init__(self, N=128, L=64.0, tau=0.2, c=1.0, eta=1.0, kappa=1.0, sigma=1.0):
        import numpy as np
        self.N  = int(N)
        self.L  = float(L)
        self.dx = self.L / self.N
        self.tau   = float(tau)
        self.c     = float(c)
        self.eta   = float(eta)
        self.kappa = float(kappa)
        self.sigma = float(sigma)

        # Grid
        x = np.linspace(0.0, self.L, self.N, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing='ij')

        # Fields
        self.mu = np.zeros((self.N, self.N, self.N), dtype=np.float64)
        self.Jx = np.zeros_like(self.mu)
        self.Jy = np.zeros_like(self.mu)
        self.Jz = np.zeros_like(self.mu)

        # Normalized 3D Gaussian kernel (torus-aware via roll in deposit)
        r2 = (self.X - self.L/2.0)**2 + (self.Y - self.L/2.0)**2 + (self.Z - self.L/2.0)**2
        s2 = (self.sigma * self.dx)**2
        g  = np.exp(-0.5 * r2 / max(s2, 1e-300))
        self.kernel = g / (g.sum() * self.dx**3)  # unit integral

        self.masses = []  # list of (x,y,z,m)

        # Initialize JAX functions if available
        # Note: JAX is disabled by default as it's slower for single-point interpolations
        # due to array conversion overhead. Enable only for vectorized operations.
        self.use_jax = False
        self._init_jax_functions()

    # ----------------- mass configuration -----------------

    def set_masses(self, quads):
        """quads: iterable of (x, y, z, m) in physical units."""
        self.masses = [(float(x) % self.L, float(y) % self.L,
                        float(z) % self.L, float(m)) for (x,y,z,m) in quads]

    # ----------------- periodic trilinear sampling -----------------

    def sample_array(self, arr, x, y, z):
        """
        Trilinear-sample a 3D grid `arr` (N×N×N) at physical coords (x,y,z)
        with periodic boundary conditions.
        """
        # Use JAX version if available
        if self.use_jax and JAX_AVAILABLE:
            return self.sample_array_jax(arr, x, y, z)

        # Fallback to NumPy version
        import numpy as np
        N = self.N; dx = self.dx

        ux = (x / dx) % N; uy = (y / dx) % N; uz = (z / dx) % N
        i0 = int(np.floor(ux)) % N
        j0 = int(np.floor(uy)) % N
        k0 = int(np.floor(uz)) % N
        i1 = (i0 + 1) % N
        j1 = (j0 + 1) % N
        k1 = (k0 + 1) % N
        fx = float(ux - np.floor(ux))
        fy = float(uy - np.floor(uy))
        fz = float(uz - np.floor(uz))

        c000 = arr[i0, j0, k0]; c100 = arr[i1, j0, k0]
        c010 = arr[i0, j1, k0]; c110 = arr[i1, j1, k0]
        c001 = arr[i0, j0, k1]; c101 = arr[i1, j0, k1]
        c011 = arr[i0, j1, k1]; c111 = arr[i1, j1, k1]

        c00 = c000 * (1.0 - fx) + c100 * fx
        c10 = c010 * (1.0 - fx) + c110 * fx
        c01 = c001 * (1.0 - fx) + c101 * fx
        c11 = c011 * (1.0 - fx) + c111 * fx

        c0 = c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy) + c11 * fy

        return float(c0 * (1.0 - fz) + c1 * fz)

    def sample_vec3(self, ax_grid, ay_grid, az_grid, x, y, z):
        """Sample three aligned grids at (x,y,z)."""
        return (self.sample_array(ax_grid, x, y, z),
                self.sample_array(ay_grid, x, y, z),
                self.sample_array(az_grid, x, y, z))

    def sample_mu(self, x, y, z):
        """Sample μ at (x,y,z)."""
        return float(self.sample_array(self.mu, x, y, z))

    def sample_J(self, x, y, z):
        """Sample J=(Jx,Jy,Jz) at (x,y,z)."""
        return (self.sample_array(self.Jx, x, y, z),
                self.sample_array(self.Jy, x, y, z),
                self.sample_array(self.Jz, x, y, z))

    def sample_grad_trilinear(self, arr, x, y, z):
        """
        Gradient of a trilinearly-interpolated 3D grid `arr` at (x,y,z) with periodic BCs.
        Returns (∂/∂x, ∂/∂y, ∂/∂z) in physical units.
        """
        # Use JAX version if available
        if self.use_jax and JAX_AVAILABLE:
            return self.grad_array_jax(arr, x, y, z)

        # Fallback to NumPy version
        import numpy as np
        N = self.N; dx = self.dx

        ux = (x / dx) % N; uy = (y / dx) % N; uz = (z / dx) % N
        i0 = int(np.floor(ux)) % N
        j0 = int(np.floor(uy)) % N
        k0 = int(np.floor(uz)) % N
        i1 = (i0 + 1) % N
        j1 = (j0 + 1) % N
        k1 = (k0 + 1) % N
        fx = float(ux - np.floor(ux))
        fy = float(uy - np.floor(uy))
        fz = float(uz - np.floor(uz))

        p000 = arr[i0, j0, k0]; p100 = arr[i1, j0, k0]
        p010 = arr[i0, j1, k0]; p110 = arr[i1, j1, k0]
        p001 = arr[i0, j0, k1]; p101 = arr[i1, j0, k1]
        p011 = arr[i0, j1, k1]; p111 = arr[i1, j1, k1]

        dphidx = (
            (p100 - p000) * (1 - fy) * (1 - fz) +
            (p110 - p010) * (    fy) * (1 - fz) +
            (p101 - p001) * (1 - fy) * (    fz) +
            (p111 - p011) * (    fy) * (    fz)
        ) / dx

        dphidy = (
            (p010 - p000) * (1 - fx) * (1 - fz) +
            (p110 - p100) * (    fx) * (1 - fz) +
            (p011 - p001) * (1 - fx) * (    fz) +
            (p111 - p101) * (    fx) * (    fz)
        ) / dx

        dphidz = (
            (p001 - p000) * (1 - fx) * (1 - fy) +
            (p101 - p100) * (    fx) * (1 - fy) +
            (p011 - p010) * (1 - fx) * (    fy) +
            (p111 - p110) * (    fx) * (    fy)
        ) / dx

        return float(dphidx), float(dphidy), float(dphidz)

    def grad_array(self, arr, x, y, z):
        """
        Gradient of a trilinearly-interpolated 3D grid `arr` at (x,y,z) with periodic BCs.
        Returns (∂/∂x, ∂/∂y, ∂/∂z) in physical units.
        """
        import numpy as np
        N = self.N; dx = self.dx

        ux = (x / dx) % N; uy = (y / dx) % N; uz = (z / dx) % N
        i0 = int(np.floor(ux)) % N
        j0 = int(np.floor(uy)) % N
        k0 = int(np.floor(uz)) % N
        i1 = (i0 + 1) % N
        j1 = (j0 + 1) % N
        k1 = (k0 + 1) % N
        fx = float(ux - np.floor(ux))
        fy = float(uy - np.floor(uy))
        fz = float(uz - np.floor(uz))

        p000 = arr[i0, j0, k0]; p100 = arr[i1, j0, k0]
        p010 = arr[i0, j1, k0]; p110 = arr[i1, j1, k0]
        p001 = arr[i0, j0, k1]; p101 = arr[i1, j0, k1]
        p011 = arr[i0, j1, k1]; p111 = arr[i1, j1, k1]

        dphidx = (
            (p100 - p000) * (1 - fy) * (1 - fz) +
            (p110 - p010) * (    fy) * (1 - fz) +
            (p101 - p001) * (1 - fy) * (    fz) +
            (p111 - p011) * (    fy) * (    fz)
        ) / dx

        dphidy = (
            (p010 - p000) * (1 - fx) * (1 - fz) +
            (p110 - p100) * (    fx) * (1 - fz) +
            (p011 - p001) * (1 - fx) * (    fz) +
            (p111 - p101) * (    fx) * (    fz)
        ) / dx

        dphidz = (
            (p001 - p000) * (1 - fx) * (1 - fy) +
            (p101 - p100) * (    fx) * (1 - fy) +
            (p011 - p010) * (1 - fx) * (    fy) +
            (p111 - p110) * (    fx) * (    fy)
        ) / dx

        return float(dphidx), float(dphidy), float(dphidz)

    # ----------------- JAX-optimized versions -----------------

    def _init_jax_functions(self):
        """Initialize JAX-compiled functions for trilinear interpolation."""
        if not JAX_AVAILABLE:
            return

        @jit
        def _sample_array_jax(arr, x, y, z, N, dx):
            """JAX-compiled trilinear interpolation."""
            ux = (x / dx) % N
            uy = (y / dx) % N
            uz = (z / dx) % N

            i0 = jnp.floor(ux).astype(jnp.int32) % N
            j0 = jnp.floor(uy).astype(jnp.int32) % N
            k0 = jnp.floor(uz).astype(jnp.int32) % N
            i1 = (i0 + 1) % N
            j1 = (j0 + 1) % N
            k1 = (k0 + 1) % N

            fx = ux - jnp.floor(ux)
            fy = uy - jnp.floor(uy)
            fz = uz - jnp.floor(uz)

            c000 = arr[i0, j0, k0]; c100 = arr[i1, j0, k0]
            c010 = arr[i0, j1, k0]; c110 = arr[i1, j1, k0]
            c001 = arr[i0, j0, k1]; c101 = arr[i1, j0, k1]
            c011 = arr[i0, j1, k1]; c111 = arr[i1, j1, k1]

            c00 = c000 * (1.0 - fx) + c100 * fx
            c10 = c010 * (1.0 - fx) + c110 * fx
            c01 = c001 * (1.0 - fx) + c101 * fx
            c11 = c011 * (1.0 - fx) + c111 * fx

            c0 = c00 * (1.0 - fy) + c10 * fy
            c1 = c01 * (1.0 - fy) + c11 * fy

            return c0 * (1.0 - fz) + c1 * fz

        # Vectorized version for multiple points
        @jit
        def _sample_array_vectorized(arr, points, N, dx):
            """JAX-compiled vectorized trilinear interpolation for multiple points."""
            x, y, z = points[:, 0], points[:, 1], points[:, 2]

            ux = (x / dx) % N
            uy = (y / dx) % N
            uz = (z / dx) % N

            i0 = jnp.floor(ux).astype(jnp.int32) % N
            j0 = jnp.floor(uy).astype(jnp.int32) % N
            k0 = jnp.floor(uz).astype(jnp.int32) % N
            i1 = (i0 + 1) % N
            j1 = (j0 + 1) % N
            k1 = (k0 + 1) % N

            fx = ux - jnp.floor(ux)
            fy = uy - jnp.floor(uy)
            fz = uz - jnp.floor(uz)

            c000 = arr[i0, j0, k0]; c100 = arr[i1, j0, k0]
            c010 = arr[i0, j1, k0]; c110 = arr[i1, j1, k0]
            c001 = arr[i0, j0, k1]; c101 = arr[i1, j0, k1]
            c011 = arr[i0, j1, k1]; c111 = arr[i1, j1, k1]

            c00 = c000 * (1.0 - fx) + c100 * fx
            c10 = c010 * (1.0 - fx) + c110 * fx
            c01 = c001 * (1.0 - fx) + c101 * fx
            c11 = c011 * (1.0 - fx) + c111 * fx

            c0 = c00 * (1.0 - fy) + c10 * fy
            c1 = c01 * (1.0 - fy) + c11 * fy

            return c0 * (1.0 - fz) + c1 * fz

        @jit
        def _grad_array_jax(arr, x, y, z, N, dx):
            """JAX-compiled gradient of trilinear interpolation."""
            ux = (x / dx) % N
            uy = (y / dx) % N
            uz = (z / dx) % N

            i0 = jnp.floor(ux).astype(jnp.int32) % N
            j0 = jnp.floor(uy).astype(jnp.int32) % N
            k0 = jnp.floor(uz).astype(jnp.int32) % N
            i1 = (i0 + 1) % N
            j1 = (j0 + 1) % N
            k1 = (k0 + 1) % N

            fx = ux - jnp.floor(ux)
            fy = uy - jnp.floor(uy)
            fz = uz - jnp.floor(uz)

            p000 = arr[i0, j0, k0]; p100 = arr[i1, j0, k0]
            p010 = arr[i0, j1, k0]; p110 = arr[i1, j1, k0]
            p001 = arr[i0, j0, k1]; p101 = arr[i1, j0, k1]
            p011 = arr[i0, j1, k1]; p111 = arr[i1, j1, k1]

            dphidx = (
                (p100 - p000) * (1 - fy) * (1 - fz) +
                (p110 - p010) * (    fy) * (1 - fz) +
                (p101 - p001) * (1 - fy) * (    fz) +
                (p111 - p011) * (    fy) * (    fz)
            ) / dx

            dphidy = (
                (p010 - p000) * (1 - fx) * (1 - fz) +
                (p110 - p100) * (    fx) * (1 - fz) +
                (p011 - p001) * (1 - fx) * (    fz) +
                (p111 - p101) * (    fx) * (    fz)
            ) / dx

            dphidz = (
                (p001 - p000) * (1 - fx) * (1 - fy) +
                (p101 - p100) * (    fx) * (1 - fy) +
                (p011 - p010) * (1 - fx) * (    fy) +
                (p111 - p110) * (    fx) * (    fy)
            ) / dx

            return dphidx, dphidy, dphidz

        self._sample_array_jax = _sample_array_jax
        self._grad_array_jax = _grad_array_jax
        self._sample_array_vectorized = _sample_array_vectorized
        self.use_jax = True

    def sample_array_jax(self, arr, x, y, z):
        """JAX-optimized version of sample_array."""
        if self.use_jax and JAX_AVAILABLE:
            arr_jax = jnp.asarray(arr)
            result = self._sample_array_jax(arr_jax, x, y, z, self.N, self.dx)
            return float(result)
        else:
            return self.sample_array(arr, x, y, z)

    def grad_array_jax(self, arr, x, y, z):
        """JAX-optimized version of grad_array."""
        if self.use_jax and JAX_AVAILABLE:
            arr_jax = jnp.asarray(arr)
            dphidx, dphidy, dphidz = self._grad_array_jax(arr_jax, x, y, z, self.N, self.dx)
            return float(dphidx), float(dphidy), float(dphidz)
        else:
            return self.grad_array(arr, x, y, z)

    # ----------------- finite-difference helpers -----------------

    def _shift(self, A, i, j, k):
        import numpy as np
        return np.roll(np.roll(np.roll(A, i, axis=0), j, axis=1), k, axis=2)

    def grad_mu(self):
        """Central differences for ∇μ (used in dynamics)."""
        dμdx = (self._shift(self.mu,-1,0,0) - self._shift(self.mu,+1,0,0)) / (2*self.dx)
        dμdy = (self._shift(self.mu,0,-1,0) - self._shift(self.mu,0,+1,0)) / (2*self.dx)
        dμdz = (self._shift(self.mu,0,0,-1) - self._shift(self.mu,0,0,+1)) / (2*self.dx)
        return dμdx, dμdy, dμdz

    def spectral_grad_mu(self):
        """FFT gradient of μ (smooth, useful for diagnostics)."""
        import numpy as np
        k = 2*np.pi*np.fft.fftfreq(self.N, d=self.dx)
        KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
        MU_k = np.fft.fftn(self.mu)
        gx = np.fft.ifftn(1j*KX*MU_k).real
        gy = np.fft.ifftn(1j*KY*MU_k).real
        gz = np.fft.ifftn(1j*KZ*MU_k).real
        return gx, gy, gz

    def div_J(self):
        """Central differences for ∇·J."""
        dJxdx = (self._shift(self.Jx,-1,0,0) - self._shift(self.Jx,+1,0,0)) / (2*self.dx)
        dJydy = (self._shift(self.Jy,0,-1,0) - self._shift(self.Jy,0,+1,0)) / (2*self.dx)
        dJzdz = (self._shift(self.Jz,0,0,-1) - self._shift(self.Jz,0,0,+1)) / (2*self.dx)
        return dJxdx + dJydy + dJzdz

    # ----------------- source deposition & time step -----------------

    def deposit(self):
        """
        Deposit sources S(x) from mass list using torus-aware Gaussian kernel.
        Each mass contributes η·m times the normalized kernel centered at (x,y,z).
        """
        import numpy as np
        S = np.zeros_like(self.mu)
        if not self.masses:
            return S
        i_c = self.N // 2
        for (xi, yi, zi, m) in self.masses:
            lam = self.eta * m
            i0 = int(np.floor(xi / self.dx)) % self.N
            j0 = int(np.floor(yi / self.dx)) % self.N
            k0 = int(np.floor(zi / self.dx)) % self.N
            # roll kernel so its center maps to the target cell
            g = np.roll(np.roll(np.roll(self.kernel,
                                        i0 - i_c, axis=0),
                                        j0 - i_c, axis=1),
                                        k0 - i_c, axis=2)
            S += lam * g
        return S

    def step(self, dt):
        """
        Advance one time step:
          1) update J semi-implicitly with -c^2 ∇μ
          2) update μ with -∇·J + S
        """
        import numpy as np
        dμdx, dμdy, dμdz = self.grad_mu()
        Ax = - self.c*self.c * dμdx
        Ay = - self.c*self.c * dμdy
        Az = - self.c*self.c * dμdz

        fac = 1.0 / (1.0 + dt/self.tau)
        self.Jx = fac * (self.Jx + (dt/self.tau)*Ax)
        self.Jy = fac * (self.Jy + (dt/self.tau)*Ay)
        self.Jz = fac * (self.Jz + (dt/self.tau)*Az)

        S = self.deposit()
        divJ = self.div_J()
        self.mu = self.mu + dt * (-divJ + S)

        # numerical hygiene
        self.mu = np.nan_to_num(self.mu, nan=0.0, posinf=0.0, neginf=0.0)
        self.Jx = np.nan_to_num(self.Jx, nan=0.0, posinf=0.0, neginf=0.0)
        self.Jy = np.nan_to_num(self.Jy, nan=0.0, posinf=0.0, neginf=0.0)
        self.Jz = np.nan_to_num(self.Jz, nan=0.0, posinf=0.0, neginf=0.0)

# --------------------------- Helmholtz projection ---------------------------

def project_curl_free_J(field):
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

def measure_slope_J(field, rmin, rmax, Nr=36, cx=None, cy=None, cz=None, ndir=512):
    """
    Robust slope: spherical shell-average of |J| at radii in [rmin, rmax].
    Uses Fibonacci-sphere directions to reduce anisotropy. Returns slope p
    in |J|(r) ~ r^p and saves field_slope_3d.png.
    """
    import numpy as np
    L = field.L
    if cx is None or cy is None or cz is None:
        cx = cy = cz = L/2.0

    # Fibonacci-sphere directions (deterministic, uniform-ish)
    i = np.arange(ndir, dtype=float)
    ga = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    z  = 1.0 - 2.0*(i + 0.5)/ndir
    rxy = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    th = ga * i
    ux = rxy * np.cos(th); uy = rxy * np.sin(th); uz = z

    rr = np.logspace(np.log10(rmin), np.log10(rmax), Nr)
    vals = np.empty(Nr, dtype=float)

    for k, r in enumerate(rr):
        acc = 0.0
        for j in range(ndir):
            x = (cx + r*ux[j]) % L
            y = (cy + r*uy[j]) % L
            zc = (cz + r*uz[j]) % L
            Jx, Jy, Jz = field.sample_J(x, y, zc)
            acc += np.sqrt(Jx*Jx + Jy*Jy + Jz*Jz)
        vals[k] = acc / ndir + 1e-300  # average |J|

    p = np.polyfit(np.log(rr), np.log(vals), 1)[0]

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.6,4.4), dpi=140)
    plt.loglog(rr, vals, 'o-', label='⟨|J|⟩_sphere(r)')
    plt.loglog(rr, rr**(-2), '--', label='r^-2 (ref, up to scale)')
    plt.xlabel("r"); plt.ylabel("⟨|J|⟩"); plt.legend()
    plt.title(f"3D log⟨|J|⟩ slope ≈ {p:.3f} (target -2.000)")
    plt.tight_layout(); plt.savefig("field_slope_3d.png"); plt.close()

    print(f"[slope_3d] (spherical) fitted slope p ≈ {p:.6f}  (target -2.000)")
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

# --------------------------- helpers: geometry ---------------------------

def plot_periodic_path(ax, X, Y, L):
    X = np.asarray(X); Y = np.asarray(Y)
    if X.size < 2:
        ax.plot(X, Y, lw=1.2); return
    jump = (np.abs(np.diff(X)) > 0.5*L) | (np.abs(np.diff(Y)) > 0.5*L)
    s = 0
    for i,c in enumerate(jump,1):
        if c: ax.plot(X[s:i], Y[s:i], lw=1.2); s = i
    ax.plot(X[s:], Y[s:], lw=1.2)

def min_image(d, L):
    return (d + 0.5*L) % L - 0.5*L

def min_image_arr(d, L):
    return (d + 0.5*L) % L - 0.5*L

def radial_profile_mu(field, cx, cy, cz, nbins=1024):
    """
    Build 1D radial profile mu_prof(r) by shell-averaging μ around (cx,cy,cz)
    under periodic BCs. Returns (r_centers, mu_prof) with r in [0, L/2).
    """
    L = field.L
    dx = min_image_arr(field.X - cx, L)
    dy = min_image_arr(field.Y - cy, L)
    dz = min_image_arr(field.Z - cz, L)
    r  = np.sqrt(dx*dx + dy*dy + dz*dz)

    rmax = 0.5*L
    nb   = int(max(128, min(nbins, field.N*4)))
    edges = np.linspace(0.0, rmax, nb+1, dtype=float)

    idx = np.searchsorted(edges, r, side="right") - 1
    idx = np.clip(idx, 0, nb-1)

    sums   = np.bincount(idx.ravel(), weights=field.mu.ravel(), minlength=nb)
    counts = np.bincount(idx.ravel(), minlength=nb)
    prof   = sums / np.maximum(1, counts)

    r_centers = 0.5*(edges[:-1] + edges[1:])
    return r_centers, prof

# --------------------------- field snapshot I/O ---------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def _save_field(prefix: str, mu, J):
    p = Path(prefix)
    p.parent.mkdir(parents=True, exist_ok=True)
    f_mu = p.with_suffix('.mu.npy')
    f_jx = p.with_suffix('.Jx.npy')
    f_jy = p.with_suffix('.Jy.npy')
    f_jz = p.with_suffix('.Jz.npy')
    np.save(f_mu, mu)
    np.save(f_jx, J[0]); np.save(f_jy, J[1]); np.save(f_jz, J[2])
    manifest = {
        "created": int(time.time()),
        "files": {
            "mu": {"path": str(f_mu), "sha256": _sha256(f_mu)},
            "Jx": {"path": str(f_jx), "sha256": _sha256(f_jx)},
            "Jy": {"path": str(f_jy), "sha256": _sha256(f_jy)},
            "Jz": {"path": str(f_jz), "sha256": _sha256(f_jz)},
        }
    }
    man = p.with_suffix('.manifest.json')
    with man.open('w') as f:
        json.dump(manifest, f, indent=2)
    print(f'[field] saved snapshot prefix="{prefix}"')

def _load_field(prefix: str):
    p = Path(prefix)
    mu = np.load(p.with_suffix('.mu.npy'))
    Jx = np.load(p.with_suffix('.Jx.npy'))
    Jy = np.load(p.with_suffix('.Jy.npy'))
    Jz = np.load(p.with_suffix('.Jz.npy'))
    print(f'[field] loaded snapshot prefix="{prefix}"')
    return mu, (Jx, Jy, Jz)

def _rel_l2(a, b):
    num = np.linalg.norm(a - b)
    den = max(1e-12, np.linalg.norm(a))
    return num / den

# --------------------------- orbit (freeze field) ---------------------------

def _robust_loglog_slope(r, Fr, rmin=None, rmax=None, nbins=24, trim=0.10):
    import numpy as np
    r = np.asarray(r); Fr = np.asarray(Fr)
    m = np.isfinite(r) & np.isfinite(Fr) & (Fr > 0)
    if not m.any(): return np.nan, np.nan

    if rmin is None: rmin = np.nanpercentile(r[m], 5)
    if rmax is None: rmax = np.nanpercentile(r[m], 95)
    m &= (r >= rmin) & (r <= rmax)
    if m.sum() < 10: return np.nan, np.nan

    # log-binning
    edges = np.logspace(np.log10(r[m].min()), np.log10(r[m].max()), nbins + 1)
    xb, yb = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = m & (r >= lo) & (r < hi)
        if sel.any():
            xb.append(np.log(np.median(r[sel])))
            yb.append(np.log(np.median(Fr[sel])))
    xb = np.asarray(xb); yb = np.asarray(yb)
    if xb.size < 5: return np.nan, np.nan

    # trim extremes
    k = max(1, int(trim * xb.size))
    order = np.argsort(xb)
    xb = xb[order][k:xb.size - k]
    yb = yb[order][k:yb.size - k]
    if xb.size < 3: return np.nan, np.nan

    A = np.vstack([xb, np.ones_like(xb)]).T
    slope, intercept = np.linalg.lstsq(A, yb, rcond=None)[0]

    # Jackknife error
    if xb.size > 5:
        sls = []
        for i in range(xb.size):
            Ai = np.vstack([np.delete(xb, i), np.ones(xb.size - 1)]).T
            bi = np.delete(yb, i)
            sls.append(np.linalg.lstsq(Ai, bi, rcond=None)[0][0])
        serr = float(np.sqrt(np.var(sls, ddof=1)))
    else:
        serr = np.nan
    return float(slope), float(serr)

def _estimate_precession(theta, r):
    import numpy as np
    # unwrap to remove 2π jumps
    theta = np.unwrap(np.asarray(theta))
    invr = 1.0 / np.clip(np.asarray(r), 1e-9, None)
    m = np.isfinite(theta) & np.isfinite(invr)
    theta, invr = theta[m], invr[m]
    if invr.size < 5: return float('nan')
    # simple peak pick on 1/r for pericenters
    pk = np.r_[False, (invr[1:-1] > invr[:-2]) & (invr[1:-1] > invr[2:]), False]
    tpk = theta[pk]
    if tpk.size < 2: return float('nan')
    dth = np.diff(tpk)
    return float(np.degrees(np.median(dth)))

def compute_diagnostics(traj):
    import numpy as np
    T = len(traj['t'])
    h = slice(T//2, T)                # use second half
    E, L = traj['E'][h], traj['L'][h]
    dE_over_E = (np.nanpercentile(E,95) - np.nanpercentile(E,5)) / np.nanmean(E)
    dL_over_L = (np.nanpercentile(L,95) - np.nanpercentile(L,5)) / np.nanmean(L)

    r = traj['r'][h]
    Fr = np.abs(traj['Fr'][h])

    slope, slope_err = _robust_loglog_slope(r, Fr, nbins=24, trim=0.10)

    prec = _estimate_precession(traj['theta'], traj['r'])

    return dict(dE_over_E=float(dE_over_E),
                dL_over_L=float(dL_over_L),
                slope_Fr=float(slope),
                slope_err=float(slope_err),
                precession_deg_per_orbit=float(prec))

def test_orbit_3d(N=128, L=64.0, masses=[(32,32,32,1.0)], eta=1.0, kappa=1.0, tau=0.2, c=1.0,
                  dt=0.02, warm=6000, steps=8000, r0=10.0, vtheta=0.32, sigma=1.0,
                  project_curl_free=False, orbit_from_potential=False,
                  symmetrize_readout=False, debug_orbit=False,
                  save_field=None, load_field=None, warm_check_k=200, warm_stop_eps=1e-4):

    fld = CollapseField3D(N=N, L=L, tau=tau, c=c, eta=eta, kappa=kappa, sigma=sigma)
    fld.set_masses(masses)

    # ---- Load cached field or warm to steady-state ----
    warm_done = 0
    if load_field:
        mu, (Jx, Jy, Jz) = _load_field(load_field)
        fld.mu[:] = mu
        fld.Jx[:] = Jx; fld.Jy[:] = Jy; fld.Jz[:] = Jz
    else:
        mu_prev = None
        J_prev  = None
        for t in range(int(warm)):
            fld.step(dt)
            warm_done += 1
            if warm_check_k and (t % warm_check_k == 0) and t > 0:
                if mu_prev is not None and J_prev is not None:
                    r_mu = _rel_l2(fld.mu, mu_prev)
                    r_J  = ( _rel_l2(fld.Jx, J_prev[0]) +
                             _rel_l2(fld.Jy, J_prev[1]) +
                             _rel_l2(fld.Jz, J_prev[2]) ) / 3.0
                    print(f'[warm] t={t} r_mu={r_mu:.3e} r_J={r_J:.3e}')
                    if r_mu < warm_stop_eps and r_J < warm_stop_eps:
                        print(f'[warm] steady-state reached at t={t}')
                        break
                mu_prev = fld.mu.copy()
                J_prev  = (fld.Jx.copy(), fld.Jy.copy(), fld.Jz.copy())
        if save_field:
            _save_field(save_field, fld.mu, (fld.Jx, fld.Jy, fld.Jz))

    # Optional: project J for J-readout (only relevant if we use J)
    if project_curl_free and not orbit_from_potential:
        project_curl_free_J(fld)
        print("[curl_free] J projected before orbit readout")

    cx,cy,cz,_ = masses[0]

    # ---- Acceleration & potential samplers ----
    if orbit_from_potential:
        if symmetrize_readout:
            # PURE RADIAL (MONOPOLE) READOUT: exactly central
            r_bins, mu_prof = radial_profile_mu(fld, cx, cy, cz, nbins=1024)
            dmu = np.gradient(mu_prof, r_bins, edge_order=2)

            def ar_of_r(rr):
                rr_c = np.clip(rr, r_bins[0], r_bins[-1])
                dmu_r = np.interp(rr_c, r_bins, dmu)
                # a = + κ c^2 ∇μ = κ c^2 (dμ/dr) r̂
                return fld.kappa * (fld.c**2) * dmu_r

            def phi_of_r(rr):
                rr_c = np.clip(rr, r_bins[0], r_bins[-1])
                mu_r = np.interp(rr_c, r_bins, mu_prof)
                # φ_em = - κ c^2 μ  (so a = -∇φ_em points inward)
                return -fld.kappa * (fld.c**2) * mu_r

            def accel_sample(x, y, z):
                rx = min_image(x - cx, fld.L); ry = min_image(y - cy, fld.L); rz = min_image(z - cz, fld.L)
                r  = np.sqrt(rx*rx + ry*ry + rz*rz) + 1e-30
                ar = ar_of_r(r)
                return ar*rx/r, ar*ry/r, ar*rz/r

            def phi_em(x):
                rx = min_image(x[0] - cx, fld.L); ry = min_image(x[1] - cy, fld.L); rz = min_image(x[2] - cz, fld.L)
                r  = np.sqrt(rx*rx + ry*ry + rz*rz)
                return phi_of_r(r)

            acc_title = "a = + κ c^2 ∂r μ(r) r̂ (monopole)"

        else:
            # MATCHED POTENTIAL (conservative, trilinear)
            phi_grid = -fld.kappa * (fld.c**2) * fld.mu
            acc_title = "a = -∇φ_em (trilinear matched)"

            # Calibrate sign once so acceleration is inward at the start radius r0
            test_x = (cx + r0) % L; test_y = cy % L; test_z = cz % L
            gx0, gy0, gz0 = fld.sample_grad_trilinear(phi_grid, test_x, test_y, test_z)  # ∇φ
            ax0, ay0, az0 = (-gx0, -gy0, -gz0)                                           # a = -∇φ
            rx0 = min_image(test_x - cx, L); ry0 = min_image(test_y - cy, L); rz0 = min_image(test_z - cz, L)
            r0mag = (rx0*rx0 + ry0*ry0 + rz0*rz0)**0.5 + 1e-30
            a0mag = (ax0*ax0 + ay0*ay0 + az0*az0)**0.5 + 1e-30
            cos0  = -(rx0*ax0 + ry0*ay0 + rz0*az0) / (r0mag * a0mag)
            SIGN  = 1.0 if cos0 >= 0 else -1.0

            def accel_sample(x, y, z):
                gx, gy, gz = fld.sample_grad_trilinear(phi_grid, x, y, z)  # ∇φ
                return (SIGN * -gx, SIGN * -gy, SIGN * -gz)                # a = -∇φ (inward)

            def phi_em(x):
                return fld.sample_array(phi_grid, x[0], x[1], x[2])

    else:
        # J-readout (non-conservative)
        aX = fld.kappa * fld.Jx
        aY = fld.kappa * fld.Jy
        aZ = fld.kappa * fld.Jz
        acc_title = "a = + κ J"

        def accel_sample(x, y, z):
            return fld.sample_vec3(aX, aY, aZ, x, y, z)

        def phi_em(x):
            # Energy lens consistent with conservative case
            return -fld.kappa*(fld.c**2)*fld.sample_mu(x[0], x[1], x[2])

    # initial state
    x = np.array([(cx + r0) % L, cy % L, cz % L], dtype=float)
    v = np.array([0.0, vtheta, 0.0], dtype=float)

    xs = np.zeros((steps, 3), dtype=float)
    E  = np.zeros(steps, dtype=float)
    Lnorm = np.zeros(steps, dtype=float)
    r_arr = np.zeros(steps, dtype=float)
    Fr_arr = np.zeros(steps, dtype=float)
    theta_arr = np.zeros(steps, dtype=float)
    cos_central_arr = np.zeros(steps, dtype=float)

    tau_frac_sum = 0.0
    rad_err_sum  = 0.0

    if steps > 0:
        print("[orbit] min_image seam-aware L enabled")

    for i in range(steps):
        ax, ay, az = accel_sample(x[0], x[1], x[2])
        a = np.array([ax, ay, az], dtype=float)

        # leapfrog half-step velocity (use in L and energy)
        v_half = v + 0.5*dt*a

        rx = min_image(x[0] - cx, L)
        ry = min_image(x[1] - cy, L)
        rz = min_image(x[2] - cz, L)
        rvec = np.array([rx, ry, rz], dtype=float)

        Lvec = np.cross(rvec, v_half)
        Lnorm[i] = np.linalg.norm(Lvec)

        E[i] = 0.5*np.dot(v_half, v_half) + phi_em(x)

        rmag = np.linalg.norm(rvec) + 1e-30
        amag = np.linalg.norm(a) + 1e-30

        # Store r, Fr (radial force), and theta for diagnostics
        r_arr[i] = rmag
        Fr_arr[i] = -np.dot(rvec, a) / rmag  # radial component of force (negative = inward)
        theta_arr[i] = np.arctan2(ry, rx)

        # ADD THIS LINE so diagnostics can mask curl-heavy segments
        cos_central_arr[i] = -np.dot(rvec, a) / (rmag * amag)  # in [-1,1]; 1 = perfectly central inward

        tau_mag = np.linalg.norm(np.cross(rvec, a))
        tau_frac = tau_mag / (rmag*amag)          # -> 0 if central
        cos_central = -np.dot(rvec, a) / (rmag*amag)
        if debug_orbit and i < 3:
            print(f"[debug] step={i:04d}  cos_central={cos_central:+.4f}  "
                  f"rmag={rmag:.3e}  amag={amag:.3e}")
        rad_err = 1.0 - np.clip(cos_central, -1.0, 1.0)
        tau_frac_sum += tau_frac
        rad_err_sum  += rad_err

        x = (x + dt*v_half) % L

        ax2, ay2, az2 = accel_sample(x[0], x[1], x[2])
        a2 = np.array([ax2, ay2, az2], dtype=float)
        v  = v_half + 0.5*dt*a2

        xs[i] = x

    t = np.arange(steps)*dt
    dE = (E - E[0]) / max(1e-12, abs(E[0]))
    dL = (Lnorm - Lnorm[0]) / max(1e-12, abs(Lnorm[0]))

    fig, axp = plt.subplots(figsize=(5.4,5.0), dpi=140)
    plot_periodic_path(axp, xs[:,0], xs[:,1], L)
    axp.set_aspect("equal", adjustable="box"); axp.set_xlim(0, L); axp.set_ylim(0, L)
    axp.set_xlabel("x"); axp.set_ylabel("y")
    axp.set_title(f"Emergent orbit (XY projection) under {acc_title}")
    plt.tight_layout(); plt.savefig("orbit_emergent_3d.png"); plt.close(fig)

    plt.figure(figsize=(5.6,3.6), dpi=140)
    plt.plot(t, dE); plt.axhline(0, color='k', lw=0.8, alpha=0.5)
    plt.xlabel("time"); plt.ylabel("(E/E0)-1"); plt.title("Energy drift (emergent 3D)")
    plt.tight_layout(); plt.savefig("energy_emergent_3d.png"); plt.close()

    plt.figure(figsize=(5.6,3.6), dpi=140)
    plt.plot(t, dL); plt.axhline(0, color='k', lw=0.8, alpha=0.5)
    plt.xlabel("time"); plt.ylabel("(||L||/||L||0)-1"); plt.title("Angular momentum drift (emergent 3D)")
    plt.tight_layout(); plt.savefig("angmom_emergent_3d.png"); plt.close()

    mean_tau_frac = tau_frac_sum / max(1, steps)
    mean_rad_err  = rad_err_sum  / max(1, steps)
    mean_abs_dE   = float(np.mean(np.abs(dE)))
    mean_abs_dL   = float(np.mean(np.abs(dL)))

    # Human-friendly summaries
    print(f"[orbit_3d] mean |dE/E0| ≈ {mean_abs_dE:.3e}, mean |d||L||/||L||0| ≈ {mean_abs_dL:.3e}")
    print(f"[orbit_3d] centrality: mean_tau_frac={mean_tau_frac:.3e}, radiality_err={mean_rad_err:.3e}")

    # Canonical machine-parseable line (used by sweep/analyzers)
    print(f"[orbit_3d] vtheta={float(vtheta):.6f} dE={mean_abs_dE:.6e} dL={mean_abs_dL:.6e} "
          f"rad={mean_rad_err:.6e} tau_frac={mean_tau_frac:.6e} warm_done={int(warm_done)}")

    # Compute and print METRICS for orbit sweep parser
    traj = {
        't': t,
        'E': E,
        'L': Lnorm,
        'r': r_arr,
        'Fr': Fr_arr,
        'theta': theta_arr,
        'cos_central': cos_central_arr,
        'dx': fld.dx  # Add dx for robust slope calculation
    }
    diag = compute_diagnostics(traj)
    print("METRICS|"
          f"dE_over_E={diag['dE_over_E']:.6e}|"
          f"dL_over_L={diag['dL_over_L']:.6e}|"
          f"slope_Fr={diag['slope_Fr']:.3f}|"
          f"slope_err={diag['slope_err']:.3f}|"
          f"precession_deg_per_orbit={diag['precession_deg_per_orbit']:.3f}",
          flush=True)

# --------------------------- CLI helpers ---------------------------

def parse_masses_3d(spec, L):
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
    d = {}
    if not s:
        return d
    for chunk in s.split(";"):
        c = chunk.strip()
        if not c: continue
        if "=" not in c:
            raise ValueError(f"Bad spec: '{c}' (expected key=value[,value])")
        k, v = c.split("=", 1)
        k = k.strip()
        vals = [x.strip() for x in v.split(",") if x.strip()]
        d[k] = vals
    return d

def _cartesian_dicts(grid_dict):
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
                slope = float(line.split("≈", 1)[1].split()[0]); break
            except Exception:
                pass
    return slope

# --- Robust orbit metrics parser ---
_ORBIT_RE = re.compile(
    r'\[orbit_3d\]\s+vtheta=(?P<vtheta>[-+0-9.eE]+)\s+'
    r'dE=(?P<dE>[-+0-9.eE]+)\s+dL=(?P<dL>[-+0-9.eE]+)\s+'
    r'rad=(?P<rad>[-+0-9.eE]+)\s+tau_frac=(?P<tau>[-+0-9.eE]+)'
    r'(?:\s+warm_done=(?P<warm_done>\d+))?'
)

def _extract_orbit_metrics(stdout_tail):
    """Parse canonical orbit metrics line from child stdout."""
    vtheta = dE = dL = rad = tau = warm_done = None
    status = 'UNKNOWN'
    for line in reversed(stdout_tail.splitlines()):
        m = _ORBIT_RE.search(line)
        if m:
            vtheta    = float(m['vtheta'])
            dE        = float(m['dE'])
            dL        = float(m['dL'])
            rad       = float(m['rad'])
            tau       = float(m['tau'])
            warm_done = int(m['warm_done']) if m['warm_done'] else None
            status    = 'OK'
            break
        if 'TIMEOUT' in line or 'timed out' in line.lower():
            status = 'TIMEOUT'
    return {
        "vtheta": vtheta, "dE": dE, "dL": dL,
        "rad": rad, "tau_frac": tau,
        "warm_done": warm_done, "status": status
    }

def _build_child_cmd(kind, args_dict):
    exe = sys.executable
    me  = f'"{str(Path(__file__).resolve())}"'
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
        base += (
            f' --steps {args_dict.get("steps", 10000)}'
            f' --r0 {args_dict.get("r0", 10.0)}'
            f' --vtheta {args_dict.get("vtheta", 0.32)}'
        )
    # Booleans
    if str(args_dict.get("project_curl_free", "")).lower() in ("1","true","yes","on"):
        base += " --project_curl_free"
    if kind == "orbit" and str(args_dict.get("orbit_from_potential", "")).lower() in ("1","true","yes","on"):
        base += " --orbit_from_potential"
    if kind == "orbit" and str(args_dict.get("symmetrize_readout", "")).lower() in ("1","true","yes","on"):
        base += " --symmetrize_readout"
    # NEW: field caching + warm detector passthroughs
    if args_dict.get("load_field"):
        base += f' --load_field "{args_dict.get("load_field")}"'
    if args_dict.get("save_field"):
        base += f' --save_field "{args_dict.get("save_field")}"'
    if args_dict.get("warm_check_k"):
        base += f' --warm_check_k {args_dict.get("warm_check_k")}'
    if args_dict.get("warm_stop_eps"):
        base += f' --warm_stop_eps {args_dict.get("warm_stop_eps")}'
    return base

def _job_runner(kind, job_cfg):
    cmd = _build_child_cmd(kind, job_cfg)
    res = _run_cmd(cmd, timeout=float(job_cfg.get("timeout_sec", 3600)))
    metrics = {}
    if kind == "slope":
        metrics["slope"] = _extract_slope(res["stdout"])
    else:
        m = _extract_orbit_metrics(res["stdout"])
        metrics.update({k: v for k, v in m.items() if k not in ("status",)})
        # status resolution
        if m.get("status") == "OK" and res.get("rc", 1) == 0:
            metrics["status"] = "OK"
        elif res.get("rc") == 124:
            metrics["status"] = "TIMEOUT"
        else:
            metrics["status"] = f'RC={res.get("rc")}'
    res["metrics"] = metrics
    res["params"] = {k: v for k, v in job_cfg.items() if k != "timeout_sec"}
    return res

def run_sweep(kind, grid_spec, const_spec, out_dir="sweeps",
              max_workers=None, timeout_sec=3600.0,
              pass_eps=0.05, leaderboard_n=8, save_logs=True):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    grid   = _parse_kv_list(grid_spec)
    consts = {**_parse_kv_list(const_spec)}
    for k, vals in list(consts.items()):
        if isinstance(vals, list) and len(vals) == 1:
            consts[k] = vals[0]
    consts["timeout_sec"] = timeout_sec
    jobs = [{**g, **consts} for g in _cartesian_dicts(grid)]
    if not jobs:
        print("[sweep] No jobs in grid — nothing to run."); return None

    workers = max_workers or max(1, os.cpu_count() // 2)
    print(f"[sweep] kind={kind} jobs={len(jobs)} workers={workers} out_dir={out_dir}")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut2cfg = {ex.submit(_job_runner, kind, j): j for j in jobs}
        for fut in as_completed(fut2cfg):
            r = fut.result(); results.append(r)
            ok = (r["rc"] == 0)
            extra = ""
            if kind == "slope" and r["metrics"].get("slope") is not None:
                extra = f"  slope={r['metrics']['slope']:+.6f}"
            print(("✅" if ok else "⚠️ "), r["cmd"], extra)

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
            "kind": kind, "jobs": total, "ok": ok_count,
            "pass_eps": pass_eps if kind == "slope" else None,
            "pass": pass_count if kind == "slope" else None,
            "wall_sec": wall, "throughput_jobs_per_sec": throughput,
            "mean_slope": mean_slope if kind == "slope" else None,
        },
        "grid": grid, "const": consts, "results": results,
    }
    outp = Path(out_dir) / f"{kind}_sweep_{int(time.time())}.json"
    with open(outp, "w") as f: json.dump(report, f, indent=2)
    print(f"[sweep] saved {outp}")

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
    ap.add_argument("--project_curl_free", action="store_true",
                    help="One-shot Helmholtz projection of J before readouts (slope/orbit).")
    ap.add_argument("--orbit_from_potential", action="store_true",
                    help="Use conservative field a = + κ c^2 ∇μ (via Φ_em = - κ c^2 μ).")
    ap.add_argument("--symmetrize_readout", action="store_true",
                    help="Use monopole (spherical) μ(r) for conservative readout (exactly central).")
    ap.add_argument("--debug_orbit", action="store_true",
                    help="Print first 3 cos_central debug lines in the orbit loop.")

    # Field caching & warm detector (NEW)
    ap.add_argument("--save_field", type=str, default=None,
                    help="Prefix path (no extension) to save mu/J snapshot + manifest.")
    ap.add_argument("--load_field", type=str, default=None,
                    help="Prefix path (no extension) to load mu/J snapshot; skips warm if provided.")
    ap.add_argument("--warm_check_k", type=int, default=200,
                    help="Warmup check interval (steps) for steady-state detection.")
    ap.add_argument("--warm_stop_eps", type=float, default=1e-4,
                    help="Relative L2 threshold (mu & J) to declare steady state.")

    # Sweep mode
    ap.add_argument("--sweep", choices=["slope","orbit"], help="Run a parallel parameter sweep")
    ap.add_argument("--grid", type=str, help='Grid spec: e.g. "N=160,192; L=80; sigma=0.6,1.0; tau=0.15,0.20"')
    ap.add_argument("--const", type=str, default="", help='Constants: e.g. "dt=0.05; warm=7000; rmin=5; rmax=20"')
    ap.add_argument("--out_dir", type=str, default="sweeps")
    ap.add_argument("--max_workers", type=int, default=None)
    ap.add_argument("--timeout_sec", type=float, default=3600.0)
    ap.add_argument("--pass_eps", type=float, default=0.05)
    ap.add_argument("--leaderboard_n", type=int, default=8)

    args = ap.parse_args()

    if args.sweep:
        if not args.grid:
            raise SystemExit("ERROR: --sweep requires --grid")
        run_sweep(kind=args.sweep, grid_spec=args.grid, const_spec=args.const,
                  out_dir=args.out_dir, max_workers=args.max_workers,
                  timeout_sec=args.timeout_sec, pass_eps=args.pass_eps,
                  leaderboard_n=args.leaderboard_n)
        return 0

    masses = parse_masses_3d(args.masses, args.L) if "parse_masses_3d" in globals() else args.masses
    if args.test == "slope":
        p = test_slope_3d(N=args.N, L=args.L, masses=masses, eta=args.eta, kappa=args.kappa,
                          tau=args.tau, c=args.c, dt=args.dt, warm=args.warm,
                          rmin=args.rmin, rmax=args.rmax, sigma=args.sigma,
                          project_curl_free=args.project_curl_free)
        status = "PASS" if abs(p + 2.0) <= args.pass_eps else "WARN"
        print(f"[result] {status} slope={p:+.6f} (target -2)")
    elif args.test == "orbit":
        test_orbit_3d(N=args.N, L=args.L, masses=masses, eta=args.eta, kappa=args.kappa,
                      tau=args.tau, c=args.c, dt=args.dt, warm=args.warm, steps=args.steps,
                      r0=args.r0, vtheta=args.vtheta, sigma=args.sigma,
                      project_curl_free=args.project_curl_free,
                      orbit_from_potential=args.orbit_from_potential,
                      symmetrize_readout=args.symmetrize_readout,
                      debug_orbit=args.debug_orbit,
                      save_field=args.save_field, load_field=args.load_field,
                      warm_check_k=args.warm_check_k, warm_stop_eps=args.warm_stop_eps)
    else:
        ap.print_help(); return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

