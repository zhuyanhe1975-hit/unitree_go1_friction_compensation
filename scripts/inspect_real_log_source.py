#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np


def _get0(ds: dict, key: str) -> str | None:
    if key not in ds:
        return None
    v = ds[key]
    if isinstance(v, np.ndarray) and v.dtype == object and v.size:
        return str(v.reshape(-1)[0])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Show provenance fields of runs/real_log.npz.")
    ap.add_argument("--npz", default="runs/real_log.npz")
    args = ap.parse_args()

    ds = dict(np.load(args.npz, allow_pickle=True))
    print("npz:", args.npz)
    print("len:", int(np.asarray(ds.get("t", ds.get("time", []))).reshape(-1).size))
    print("keys:", ", ".join(sorted(ds.keys())))

    parent = _get0(ds, "parent_csv") or "(missing)"
    sha = _get0(ds, "parent_csv_sha256") or "(missing)"
    created = _get0(ds, "created_at") or "(missing)"
    head = _get0(ds, "git_head") or "(missing)"
    print("parent_csv:", parent)
    print("parent_csv_sha256:", sha)
    print("created_at:", created)
    print("git_head:", head)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

