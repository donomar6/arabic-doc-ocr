#!/usr/bin/env python
"""Build the self-contained wheel for a target interpreter/platform.

Steps: download (or reuse) the matching onnxruntime wheel, vendor it into
``src/arabic_ocr/_vendor``, then build with the wheel tag of the *target*.

    python tools/build_wheel.py --target win_amd64 --python 3.10
    python tools/build_wheel.py --target manylinux_2_28_x86_64 --python 3.10
    python tools/build_wheel.py --target win_amd64 --python 3.10 --ort-wheel path/to/onnxruntime-*.whl

Adding a platform or Python version is a matter of running this with other
``--target`` / ``--python`` values (any pair onnxruntime publishes wheels for).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "src" / "arabic_ocr" / "_vendor"

# wheel platform tag -> pip --platform value(s) used to fetch onnxruntime
TARGETS = {
    "win_amd64": ["win_amd64"],
    "win_arm64": ["win_arm64"],
    "manylinux_2_28_x86_64": ["manylinux_2_28_x86_64", "manylinux_2_27_x86_64"],
    "manylinux_2_28_aarch64": ["manylinux_2_28_aarch64", "manylinux_2_27_aarch64"],
    "macosx_13_0_universal2": ["macosx_13_0_universal2"],
}


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call([str(c) for c in cmd], **kw)


def fetch_ort(target: str, py: str, version: str | None, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    spec = f"onnxruntime=={version}" if version else "onnxruntime"
    cmd = [sys.executable, "-m", "pip", "download", spec, "--only-binary=:all:", "--no-deps", "--python-version", py, "-d", dest]
    for plat in TARGETS[target]:
        cmd += ["--platform", plat]
    run(cmd)
    cp = "cp" + py.replace(".", "")
    wheels = sorted(dest.glob(f"onnxruntime-*-{cp}-{cp}-*.whl"), key=lambda p: p.stat().st_mtime)
    wheels = [w for w in wheels if any(plat.split("_")[0] in w.name for plat in TARGETS[target])]
    if not wheels:
        raise SystemExit(f"no onnxruntime wheel found for {target} / {py}")
    return wheels[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--python", default="3.10", help="target CPython version, e.g. 3.10")
    ap.add_argument("--ort-version", default=None, help="pin the onnxruntime version (default: latest for the target)")
    ap.add_argument("--ort-wheel", type=Path, default=None, help="use this onnxruntime wheel instead of downloading")
    ap.add_argument("--outdir", type=Path, default=ROOT / "dist")
    ap.add_argument("--keep-vendor", action="store_true", help="leave the vendored copy in src/ after the build")
    args = ap.parse_args(argv)

    cp = "cp" + args.python.replace(".", "")
    ort_wheel = args.ort_wheel or fetch_ort(args.target, args.python, args.ort_version, ROOT / "build" / "ort_wheels")
    if cp not in ort_wheel.name or not any(p.split("_")[0] in ort_wheel.name for p in TARGETS[args.target]):
        raise SystemExit(f"{ort_wheel.name} does not look like a {cp} wheel for {args.target}")

    run([sys.executable, ROOT / "tools" / "vendor_onnxruntime.py", "--wheel", ort_wheel, "--dest", VENDOR])

    env = dict(os.environ, ARABIC_OCR_PY_TAG=cp, ARABIC_OCR_ABI_TAG=cp, ARABIC_OCR_PLAT_TAG=args.target)
    # setuptools caches the previous target's binaries under build/lib.<plat>; a
    # stale copy would end up inside the new wheel.
    for stale in list((ROOT / "build").glob("lib*")) + list((ROOT / "build").glob("bdist*")):
        shutil.rmtree(stale, ignore_errors=True)
    for egg in (ROOT / "src").glob("*.egg-info"):
        shutil.rmtree(egg, ignore_errors=True)
    run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", args.outdir, ROOT], env=env)

    built = sorted(args.outdir.glob(f"arabic_doc_ocr-*-{cp}-{cp}-{args.target}.whl"), key=lambda p: p.stat().st_mtime)
    if not built:
        raise SystemExit("wheel not found after build")
    wheel = built[-1]
    size = wheel.stat().st_size
    print(f"\nbuilt {wheel}  ({size / 1e6:.1f} MB, {size / (1 << 20):.1f} MiB)")
    with zipfile.ZipFile(wheel) as zf:
        infos = zf.infolist()
        total = sum(i.file_size for i in infos)
        print(f"{len(infos)} files, {total / 1e6:.1f} MB uncompressed. Largest entries:")
        for i in sorted(infos, key=lambda i: -i.file_size)[:8]:
            print(f"  {i.file_size / 1e6:7.1f} MB  {i.filename}")
    if size > 100 * 1024 * 1024:
        raise SystemExit("wheel exceeds the 100 MiB PyPI limit")
    if not args.keep_vendor:
        shutil.rmtree(VENDOR / "onnxruntime", ignore_errors=True)
        shutil.rmtree(VENDOR / "licenses", ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
