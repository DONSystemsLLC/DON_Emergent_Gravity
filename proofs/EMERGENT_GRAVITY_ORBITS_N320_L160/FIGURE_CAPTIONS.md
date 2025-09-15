# Figure Panel Captions for P1 Proof

## Figure 1 — Field Kernel (log–log)
**File:** `field_slope_warm.png`

Shell-averaged radial flux ⟨|Jᵣ|⟩ vs radius on log–log axes. Barycentric centering and minimum-image distances are used. Power-law fits across interior windows yield p = -2.069 on [6,20] and p = -2.052 on [8,24], consistent with a 1/r² kernel. The Gauss-law diagnostic Jᵣr² is flat within ~±10–11% (16–84% band), indicating radius-independent flux through spherical shells.

---

## Figure 2 — Rotation Curve from Collapse Field
**File:** `rotation_curve.png`

Field-derived circular speed vₖ(r) ∝ √(r⟨|Jᵣ|⟩) (arbitrary scale) compared to the best-orbit tangential speed at r₀ ≈ 20. The curves match at r₀ by a single scale factor, demonstrating that the collapse-adjacency field supports Kepler-like motion without a hand-inserted gravitational potential.

---

## Figure 3 — Orbit Conservation (Production Run)
**Files:** `energy_emergent_3d.png`, `angmom_emergent_3d.png` (if included as panel)

Time series of normalized energy and angular-momentum variations for the vθ = 0.30 production orbit. Mean |ΔE/E₀| ≲ 7.5×10⁻⁸ and |Δ‖L‖/‖L‖₀| ≈ 2.4×10⁻³ confirm high-fidelity integration in the fixed collapse field.

---

## Figure 4 — Orbit Geometry / Precession
**File:** `orbit_emergent_3d.png` (if included)

Projection of the test-particle orbit in the warmed field snapshot. Periapse-to-periapse precession is small and steady, ~0.022° per orbit, consistent with near-Keplerian dynamics in the emergent 1/r² kernel.

---

## LaTeX-Ready Captions

For manuscript preparation, use these LaTeX-formatted versions:

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{field_slope_warm.png}
\caption{\textbf{Field Kernel (log--log).} Shell-averaged radial flux $\langle|J_r|\rangle$ vs radius on log--log axes. Barycentric centering and minimum-image distances are used. Power-law fits across interior windows yield $p = -2.069$ on $[6,20]$ and $p = -2.052$ on $[8,24]$, consistent with a $1/r^2$ kernel. The Gauss-law diagnostic $J_r r^2$ is flat within $\sim\pm10$--$11\%$ (16--84\% band), indicating radius-independent flux through spherical shells.}
\label{fig:field-kernel}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{rotation_curve.png}
\caption{\textbf{Rotation Curve from Collapse Field.} Field-derived circular speed $v_c(r) \propto \sqrt{r\langle|J_r|\rangle}$ (arbitrary scale) compared to the best-orbit tangential speed at $r_0 \approx 20$. The curves match at $r_0$ by a single scale factor, demonstrating that the collapse-adjacency field supports Kepler-like motion without a hand-inserted gravitational potential.}
\label{fig:rotation-curve}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.49\textwidth]{energy_emergent_3d.png}
\includegraphics[width=0.49\textwidth]{angmom_emergent_3d.png}
\caption{\textbf{Orbit Conservation (Production Run).} Time series of normalized energy and angular-momentum variations for the $v_\theta = 0.30$ production orbit. Mean $|\Delta E/E_0| \lesssim 7.5\times10^{-8}$ and $|\Delta\|L\|/\|L\|_0| \approx 2.4\times10^{-3}$ confirm high-fidelity integration in the fixed collapse field.}
\label{fig:orbit-conservation}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{orbit_emergent_3d.png}
\caption{\textbf{Orbit Geometry / Precession.} Projection of the test-particle orbit in the warmed field snapshot. Periapse-to-periapse precession is small and steady, $\sim 0.022^\circ$ per orbit, consistent with near-Keplerian dynamics in the emergent $1/r^2$ kernel.}
\label{fig:orbit-geometry}
\end{figure}
```