from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sdk_src",
        default="/home/yhzhu/Industrial Robot/unitree_actuator_sdk",
        help="path to unitree_actuator_sdk source repo",
    )
    ap.add_argument("--out", default="runs/unitree_sdk_build", help="output folder inside this repo")
    ap.add_argument("--clean", action="store_true", help="remove existing build output first")
    args = ap.parse_args()

    sdk_src = os.path.abspath(args.sdk_src)
    out_root = os.path.abspath(args.out)
    build_dir = os.path.join(out_root, "build")
    lib_dir = os.path.join(out_root, "lib")

    if not os.path.isdir(sdk_src):
        raise SystemExit(f"SDK source not found: {sdk_src}")

    if args.clean and os.path.exists(out_root):
        shutil.rmtree(out_root)

    os.makedirs(build_dir, exist_ok=True)

    python_exe = sys.executable
    print(f"Using python: {python_exe}")
    print(f"SDK source: {sdk_src}")
    print(f"Build dir:  {build_dir}")

    cfg_cmd = [
        "cmake",
        "-S",
        sdk_src,
        "-B",
        build_dir,
        f"-DPYTHON_EXECUTABLE={python_exe}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    subprocess.check_call(cfg_cmd)

    build_cmd = ["cmake", "--build", build_dir, "--config", "Release", "-j"]
    subprocess.check_call(build_cmd)

    if not os.path.isdir(lib_dir):
        raise SystemExit(f"Expected output lib dir not found: {lib_dir}")

    so_files = [f for f in os.listdir(lib_dir) if f.startswith("unitree_actuator_sdk") and f.endswith(".so")]
    if not so_files:
        raise SystemExit(f"No unitree_actuator_sdk*.so found in {lib_dir}. Built files: {os.listdir(lib_dir)}")

    print("Built python module(s):")
    for f in sorted(so_files):
        print(f"  - {os.path.join(lib_dir, f)}")

    print("\nNext:")
    print(f'  - set `real.unitree_sdk_lib` in config.json to "{os.path.relpath(lib_dir)}"')


if __name__ == "__main__":
    main()

