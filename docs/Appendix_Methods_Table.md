# Appendix A — Numerical Performance and Resources

This table summarizes numerical performance and resource usage for the strict-orbit validation runs.

> Notes: measurements taken with `/usr/bin/time -l` on macOS; threads pinned via `OMP_NUM_THREADS`/`MKL_NUM_THREADS`. Steps/s computed as `steps / real_seconds`. Memory is the maximum resident set size (approx.).

| N (cells) | L | Δt | Steps | Time (s) | Steps/s | Max RSS (GB) | OMP | MKL | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 320 | 160 | 0.0015625 | 256000 | 50.2 | 5,100.6 |  | 1 | 1 | N320_L160_dt0.0015625.timelog |
| **—** | **—** | **—** | **—** | **—** | **—** | **—** | **—** | **—** | **—** |
| 320 | 160 | 0.0015625 | 256000 | 50.2 | 5,100.6 |  | 1 | 1 | — median — |

**Footnotes**  
1. Timings taken with BSD `/usr/bin/time -l` (real/user/sys, max RSS).  
2. Threads pinned: `OMP_NUM_THREADS`, `MKL_NUM_THREADS` (see columns).  
3. Steps/s = steps ÷ real seconds; each step is one integrator step.  
4. Memory is approximate; ru_maxrss semantics may differ across OS versions.
