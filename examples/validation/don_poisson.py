#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
don_poisson.py

3D Poisson solver for Φ from ρ using FFT (periodic box),
plus spectral ∇Φ for acceleration. Includes validation tests.
∇²Φ = 4π G ρ,  a = -∇Φ
"""

import numpy as np

def poisson_fft_periodic(rho, boxlen, G=1.0):
    rho = np.asarray(rho, dtype=np.float64)
    N = np.array(rho.shape, dtype=int)
    ks = [np.fft.fftfreq(n, d=boxlen/n) * 2*np.pi for n in N]
    kx, ky, kz = np.meshgrid(*ks, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2

    rho_k = np.fft.fftn(rho)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_k = np.where(k2>0, -4*np.pi*G * rho_k / k2, 0.0)

    phi = np.fft.ifftn(phi_k).real
    ax = -np.fft.ifftn(1j * kx * phi_k).real
    ay = -np.fft.ifftn(1j * ky * phi_k).real
    az = -np.fft.ifftn(1j * kz * phi_k).real
    return phi, (ax, ay, az)

# ----- analytic references -----

def plummer_density(x,y,z,M=1.0,a=0.05):
    r2 = x*x + y*y + z*z
    return (3*M)/(4*np.pi*a**3) * (1 + r2/a**2)**(-2.5)

def plummer_accel(x,y,z,G=1.0,M=1.0,a=0.05):
    r2 = x*x + y*y + z*z
    s  = (r2 + a*a)**(-1.5)
    return -G*M*x*s, -G*M*y*s, -G*M*z*s

def uniform_sphere_density(x,y,z,R=0.2,rho0=1.0):
    r = np.sqrt(x*x+y*y+z*z)
    return rho0 * (r<=R)

def uniform_sphere_accel(x,y,z,G=1.0,R=0.2,rho0=1.0):
    r = np.sqrt(x*x+y*y+z*z) + 1e-300
    M = (4/3)*np.pi*R**3*rho0
    a_in  = -(4*np.pi*G*rho0/3) * r
    a_out = -G*M / (r*r)
    amag  = np.where(r<=R, a_in, a_out)
    return amag*(x/r), amag*(y/r), amag*(z/r)

# ----- validation harness -----

def _grid(N, L):
    lin = np.linspace(-0.5*L, 0.5*L, N, endpoint=False)
    return np.meshgrid(lin, lin, lin, indexing="ij")

def validate_plummer(N=128, L=1.0, G=1.0, M=1.0, a=0.05):
    x,y,z = _grid(N,L)
    rho = plummer_density(x,y,z,M,a)
    _, (ax,ay,az) = poisson_fft_periodic(rho, L, G)
    axt,ayt,azt   = plummer_accel(x,y,z,G,M,a)
    r = np.sqrt(x*x+y*y+z*z)
    mask = r > (2.5*a)
    num = np.sqrt(ax[mask]**2 + ay[mask]**2 + az[mask]**2) + 1e-300
    ref = np.sqrt(axt[mask]**2 + ayt[mask]**2 + azt[mask]**2) + 1e-300
    rel = np.abs(num-ref)/ref
    return float(np.mean(rel)), float(np.max(rel))

def validate_uniform(N=128, L=1.0, G=1.0, R=0.2, rho0=1.0):
    x,y,z = _grid(N,L)
    rho = uniform_sphere_density(x,y,z,R,rho0)
    _, (ax,ay,az) = poisson_fft_periodic(rho, L, G)
    axt,ayt,azt   = uniform_sphere_accel(x,y,z,G,R,rho0)
    r = np.sqrt(x*x+y*y+z*z)
    mask = (np.abs(r-R) > 2*L/N) & (r < 0.45*L)
    num = np.sqrt(ax[mask]**2 + ay[mask]**2 + az[mask]**2) + 1e-300
    ref = np.sqrt(axt[mask]**2 + ayt[mask]**2 + azt[mask]**2) + 1e-300
    rel = np.abs(num-ref)/ref
    return float(np.mean(rel)), float(np.max(rel))

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("[poisson] validating...")
    m,M = validate_plummer()
    print(f"[plummer] mean rel err={m:.3e}  max rel err={M:.3e}")
    m,M = validate_uniform()
    print(f"[uniform] mean rel err={m:.3e}  max rel err={M:.3e}")
    print("[poisson] done.")

