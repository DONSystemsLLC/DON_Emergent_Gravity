# DON Theory — Emergent Gravity Overview

> **What this repo proves:** In 3D, *without* hard-coding Poisson or a potential, a collapse-transport memory field produces a **conserved radial flux** with far-field scaling **|J(r)| ∝ r⁻²**.  
> Test particles that read this flux move on **Kepler-like orbits** with small energy and angular-momentum drift.  
> This is the minimal kernel of Newtonian gravity, *derived* from information flow rather than assumed.

---

## 0) One-Screen Summary

**Thesis.** DON Theory treats reality as a *tiered information system* driven by dynamic entanglement entropy:

- **ϑ (Theta / substrate)** — slow stochastic layer, latent connectivity.  
- **Φ (Phi / memory)** — condensed coherence seeded from ϑ.  
- **Aᵢⱼ (adjacency)** — mutual-information network across Φ regions; spectral inflections drive collapse.  
- **Ψ (Psi / observables)** — fast effective dynamics (forces, particles, “laws”).

All tiers couple through one action: the **Dynamic Entanglement Entropy (DEE) Lagrangian**.  
Emergence and collapse are phases of the same information-driven dynamics.

**Kernel result (this repo).** Continuity + isotropy ⇒ conserved flux ⇒ **Gauss-like 1/r²**.  
Bodies reading this flux accelerate attractively. Newton’s law emerges with shape, sign, and composite strength:

$$
G = \frac{\alpha \eta}{4 \pi D}
$$

explaining universality of $G$ and predicting small, testable departures when transport is not steady.

---

## 1) Formalism

### 1.1 Tiered Architecture

$$
\vartheta \;\rightarrow\; \Phi \;\rightarrow\; A_{ij} \;\rightarrow\; \Psi
$$

- Collapse criterion: $\tfrac{d^2 \lambda_{\max}}{d\tau^2} = 0$.  
- Timescales: $\tau_\vartheta \gg \tau_\Phi \gg \tau_\Psi$.

### 1.2 DEE Lagrangian (sketch)

$$
\mathcal L = \tfrac12 (\partial_\mu \Phi)^2 - V(\Phi)
+ \alpha\,\mathcal A[\Phi,\vartheta]
- \tfrac{\kappa}{2}\left(\tfrac{d^2\lambda_{\max}}{d\tau^2}\right)^2
+ \eta\,\mathcal S[\vartheta]
$$

- $V(\Phi) = \beta \Phi^4 - \gamma \Phi^2$ (bistable).  
- $\mathcal A$: mutual-information coupling.  
- $\lambda_{\max}$: adjacency spectrum eigenvalue.  
- $\mathcal S[\vartheta]$: stochastic substrate driver.

Equation of motion (memory field):

$$
\square \Phi + \frac{dV}{d\Phi} = \alpha\,\frac{\delta \mathcal A}{\delta \Phi} + \eta\,\vartheta(x,t)
$$

### 1.3 Collapse-Transport Kernel

$$
\partial_t \mu + \nabla\cdot \mathbf J = S, \qquad
\tau \partial_t \mathbf J + \mathbf J = -c^2 \nabla \mu
$$

Steady isotropy ⇒ $4\pi r^2 J(r) = Q \;\Rightarrow\; J(r) \propto 1/r^2$.

Bodies reading flux:

$$
\mathbf a = +\kappa \mathbf J = -\nabla(\kappa c^2 \mu) \equiv -\nabla \Phi_{\rm em}
$$

Strength mapping:

$$
\nabla^2 \Phi = 4\pi G \rho, \qquad
G = \frac{\alpha \eta}{4\pi D}
$$

---

## 2) What the Repo Implements

- Discretized solver for $(\mu,\mathbf J)$ on periodic box $[0,L]^3$.  
- Source = compact kernel at box center.  
- Numerics: semi-implicit J-decay, explicit μ continuity.  
- Measurements:
  - **Slope:** log-log fit of $|J(r)|$ in far-field annulus.  
  - **Orbit:** velocity–Verlet in frozen field, seam-aware angular momentum.  

---

## 3) Reproduce Results

```bash
# slope (polished fit)
python src/don_emergent_collapse_3d.py --test slope \
  --N 192 --L 80 --sigma 1.0 --tau 0.20 \
  --dt 0.05 --warm 8000 --rmin 5 --rmax 20 \
  --masses 32,32,32,1.0 --project_curl_free

# orbit (conservative readout)
python src/don_emergent_collapse_3d.py --test orbit \
  --N 192 --L 80 --sigma 1.0 --tau 0.20 \
  --dt 0.025 --warm 8000 --steps 16000 \
  --r0 10.0 --vtheta 0.32 \
  --masses 32,32,32,1.0 --orbit_from_potential
Artifacts saved in results/proof_emergence_v1/.

4) Predictions Beyond Kernel
Lensing anomaly: +3.1% near z≈0.27 (cluster mergers).

Void adjacency resonance: wiggle in stacked void profiles.

Neutron-star echoes: ms-spaced post-merger cascades.

Quantum threshold: ΔΦ < δ_c → sharp collapse onset.

All falsifiable with near-term data (Euclid/LSST, DESI, LIGO/ET, macromolecule interferometry).

5) Repo Layout
bash
Copy code
DON_Emergent_Gravity/
├─ src/                     # solver
│   └─ don_emergent_collapse_3d.py
├─ scripts/
│   └─ reproduce_proof.sh
├─ results/
│   ├─ proof_emergence_v1/  # slope+orbit figs + run_params.json
│   └─ sweeps/              # provenance JSONs (optional)
├─ docs/
│   └─ OVERVIEW.md
├─ examples/ (optional)
│   ├─ lambda_demo/don_orbit_live_lambda.py
│   └─ validation/don_poisson.py
├─ requirements.txt
└─ README.md
6) Quick Start
bash
Copy code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/reproduce_proof.sh
7) Contributing
Multi-domain tiling with dynamic inter-block adjacency.

Spectral logging (λₘₐₓ curvature/gap).

Repro notebooks for slope/orbit.

CI smoke tests (tiny N).

Please keep core solver minimal (NumPy/Matplotlib only).


Is Newton/GR assumed? No. 1/r² + attraction derived from continuity + isotropy + flux readout.

Where does G come from? $G = \alpha\eta / (4\pi D)$.

Why GR recovered? Promote $\Phi_{\rm em}$ to retarded potential; Poisson limit appears from steady transport.

Where’s “fractal feedback”? Encoded in DEE tiered formalism. This repo proves only the gravity kernel.



DON Systems (2025). DON Emergent Gravity — 3D Collapse Proof (v1).
GitHub: https://github.com/DONSystemsLLC/DON_Emergent_Gravity
