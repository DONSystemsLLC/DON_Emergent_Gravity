import hashlib, pathlib as P

def sha256(path: P.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

for results in P.Path("proofs").rglob("RESULTS*.json"):
    bundle = results.parent
    lines = []
    for f in sorted(list(bundle.glob("*.json")) + list(bundle.glob("*.png")) + list(bundle.glob("*.csv"))):
        try:
            lines.append(f"{sha256(f)}  {f.name}")
        except Exception:
            pass
    (bundle / "SHA256SUMS.txt").write_text("\n".join(lines))
    print(f"[hash] wrote {bundle/'SHA256SUMS.txt'}")
