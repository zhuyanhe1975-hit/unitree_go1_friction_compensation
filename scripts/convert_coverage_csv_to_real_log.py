#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git_head() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _load_existing_parent(npz_path: Path) -> dict[str, Any]:
    if not npz_path.exists():
        return {}
    try:
        ds = dict(np.load(str(npz_path), allow_pickle=True))
    except Exception:
        return {}
    out: dict[str, Any] = {}
    for k in ("parent_csv", "parent_csv_sha256", "created_at", "git_head"):
        if k in ds:
            v = ds[k]
            # stored as object arrays
            if isinstance(v, np.ndarray) and v.dtype == object and v.size:
                out[k] = str(v.reshape(-1)[0])
            else:
                out[k] = v
    return out


def _append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert coverage_capture_*.csv into runs/real_log.npz-style npz with source tracking + stage_id."
    )
    ap.add_argument("--csv", required=True, help="coverage_capture CSV path")
    ap.add_argument("--out", default="runs/real_log.npz", help="output npz path (default: runs/real_log.npz)")
    ap.add_argument("--kp", type=float, default=2.0, help="kp used in capture (stored for context)")
    ap.add_argument("--kd", type=float, default=0.02, help="kd used in capture (stored for context)")
    ap.add_argument("--tau_cmd", type=float, default=0.0, help="tau command / bias in capture (usually 0)")
    ap.add_argument("--force", action="store_true", help="overwrite even if output already exists with different source")
    ap.add_argument("--no_backup", action="store_true", help="do not backup existing out npz before overwriting")
    ap.add_argument("--manifest", default="runs/real_log_manifest.jsonl", help="append-only manifest jsonl")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"csv not found: {csv_path}")

    csv_sha = _sha256_file(csv_path)
    created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    git_head = _git_head()

    # If output exists, require --force when the parent differs (or is unknown).
    existing = _load_existing_parent(out_path)
    if out_path.exists():
        same = False
        if existing.get("parent_csv_sha256", "") == csv_sha:
            same = True
        if not same and not args.force:
            msg = [
                f"refusing to overwrite existing npz without --force: {out_path}",
                f"- existing parent_csv={existing.get('parent_csv','(unknown)')}",
                f"- existing parent_sha256={existing.get('parent_csv_sha256','(unknown)')}",
                f"- new csv={str(csv_path)}",
                f"- new sha256={csv_sha}",
                "If you really want to replace it, re-run with --force.",
            ]
            raise SystemExit("\n".join(msg))

        if not args.no_backup:
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            backup = out_path.with_name(out_path.stem + f"_backup_{stamp}" + out_path.suffix)
            shutil.copy2(out_path, backup)
            print(f"backup: {backup}")

    # Parse CSV
    t_s: list[float] = []
    q: list[float] = []
    qd: list[float] = []
    tau: list[float] = []
    q_ref: list[float] = []
    qd_ref: list[float] = []
    stage_id: list[int] = []
    stage_map: dict[str, int] = {}
    stage_names: list[str] = []

    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"t_s", "stage", "q_rad", "dq_rad_s", "tau_Nm", "q_ref_rad", "dq_ref_rad_s"}
        missing = [k for k in sorted(required) if k not in (r.fieldnames or [])]
        if missing:
            raise SystemExit(f"CSV missing required columns: {missing}. got={r.fieldnames}")

        for row in r:
            st = row["stage"]
            if st not in stage_map:
                stage_map[st] = len(stage_names)
                stage_names.append(st)
            stage_id.append(stage_map[st])

            t_s.append(float(row["t_s"]))
            q.append(float(row["q_rad"]))
            qd.append(float(row["dq_rad_s"]))
            tau.append(float(row["tau_Nm"]))
            q_ref.append(float(row["q_ref_rad"]))
            qd_ref.append(float(row["dq_ref_rad_s"]))

    if not t_s:
        raise SystemExit("CSV has no rows")

    t = np.asarray(t_s, dtype=np.float64)
    q_out = np.asarray(q, dtype=np.float64)
    qd_out = np.asarray(qd, dtype=np.float64)
    tau_out = np.asarray(tau, dtype=np.float64)
    q_ref_arr = np.asarray(q_ref, dtype=np.float64)
    qd_ref_arr = np.asarray(qd_ref, dtype=np.float64)
    stage_id_arr = np.asarray(stage_id, dtype=np.int32)

    # In position-loop capture there is no explicit torque command; keep a tau_cmd channel for API compatibility.
    tau_cmd_arr = np.full_like(tau_out, float(args.tau_cmd), dtype=np.float64)

    kp_arr = np.full_like(tau_out, float(args.kp), dtype=np.float64)
    kd_arr = np.full_like(tau_out, float(args.kd), dtype=np.float64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        # Core time series (names match project expectation)
        t=t,
        q_out=q_out,
        qd_out=qd_out,
        tau_out=tau_out,
        tau_cmd=tau_cmd_arr,
        # Compatibility aliases (some older pipelines log motor-side keys)
        q_m=q_out,
        qd_m=qd_out,
        # Controller context (for v2)
        q_ref=q_ref_arr,
        qd_ref=qd_ref_arr,
        kp=kp_arr,
        kd=kd_arr,
        # Segmentation
        stage_id=stage_id_arr,
        stage_names=np.asarray(stage_names, dtype=object),
        # Provenance
        parent_csv=np.asarray([str(csv_path)], dtype=object),
        parent_csv_sha256=np.asarray([csv_sha], dtype=object),
        created_at=np.asarray([created_at], dtype=object),
        git_head=np.asarray([git_head], dtype=object),
    )
    print(f"saved: {out_path}")

    rec = {
        "created_at": created_at,
        "git_head": git_head,
        "out_npz": str(out_path),
        "csv": str(csv_path),
        "csv_sha256": csv_sha,
        "rows": int(len(t)),
        "stages": stage_names,
        "kp": float(args.kp),
        "kd": float(args.kd),
        "tau_cmd": float(args.tau_cmd),
    }
    _append_manifest(Path(args.manifest), rec)
    print(f"manifest_append: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
