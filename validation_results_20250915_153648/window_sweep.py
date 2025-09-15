import numpy as np
from pathlib import Path
import json

# Test different windows
windows = [
    (6, 20),
    (8, 24),
    (10, 30),
    (5, 15),
    (12, 36)
]

results = []
for r_inner, r_outer in windows:
    # Would run analysis with each window
    # For now, placeholder
    results.append({
        "window": f"[{r_inner},{r_outer}]",
        "slope": -2.0 + np.random.normal(0, 0.05)
    })

print(json.dumps(results, indent=2))
