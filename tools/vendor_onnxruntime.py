#!/usr/bin/env python
"""Vendor the onnxruntime Python package into ``arabic_ocr/_vendor/onnxruntime``.

The wheel for the *target* platform is extracted, trimmed to the parts needed
for CPU inference, and every ``import onnxruntime`` statement is rewritten to
the private ``arabic_ocr._vendor.onnxruntime`` namespace so the copy can never
shadow, or be shadowed by, an onnxruntime installed by the user.

Usage:
    python tools/vendor_onnxruntime.py --wheel onnxruntime-1.23.2-cp310-cp310-win_amd64.whl \
        --dest src/arabic_ocr/_vendor
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

VENDOR_NAMESPACE = "arabic_ocr._vendor.onnxruntime"

# Sub-packages of onnxruntime that are not needed for plain CPU inference.
DROP_DIRS = {"backend", "datasets", "quantization", "tools", "transformers", "training", "experimental", "__pycache__"}
DROP_FILES = {"convert_npz_to_onnx_adapter.py", "onnxruntime_collect_build_info.py"}
# The Python extension module is statically linked against the onnxruntime
# core; the standalone shared library in the wheel is only there for C/C#
# consumers. Dropping it saves ~15-22 MB. ``*_providers_shared`` is tiny and is
# kept so provider registration code paths stay intact.
DROP_BINARY_PREFIXES = ("onnxruntime.dll", "libonnxruntime.so", "libonnxruntime.dylib")

# pybind11 keeps one process-wide registry of bound C++ types per
# "internals id". Two onnxruntime builds that share an id (same pybind11
# internals version and compiler ABI) cannot be imported into one process:
# the second import fails with 'generic_type: type "..." is already
# registered!'. Giving our private copy its own id (same byte length, so the
# binary layout is untouched) lets it coexist with any onnxruntime the user
# has installed.
PYBIND_ID_OLD = b"__pybind11_internals_v"
PYBIND_ID_NEW = b"__pybind11_arabicocr_v"
assert len(PYBIND_ID_OLD) == len(PYBIND_ID_NEW)

_FROM_RE = re.compile(r"^(\s*)from\s+onnxruntime(\.[\w.]+)?\s+import\b", re.M)
_IMPORT_AS_RE = re.compile(r"^(\s*)import\s+onnxruntime(\.[\w.]+)?\s+as\s+(\w+)\s*$", re.M)
_IMPORT_RE = re.compile(r"^(\s*)import\s+onnxruntime\s*(#.*)?$", re.M)


def rewrite_imports(text: str) -> str:
    text = _FROM_RE.sub(lambda m: f"{m.group(1)}from {VENDOR_NAMESPACE}{m.group(2) or ''} import", text)
    text = _IMPORT_AS_RE.sub(lambda m: f"{m.group(1)}import {VENDOR_NAMESPACE}{m.group(2) or ''} as {m.group(3)}", text)
    text = _IMPORT_RE.sub(lambda m: f"{m.group(1)}import {VENDOR_NAMESPACE} as onnxruntime", text)
    return text


def _pe_fixup(path: Path) -> str:
    """After patching a Windows PE file: drop the (now invalid) Authenticode
    signature and recompute the PE checksum so the file is internally
    consistent. Pure Python, no pefile dependency."""
    import struct  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    data = bytearray(path.read_bytes())
    if data[:2] != b"MZ":
        return "not a PE file"
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if data[e_lfanew : e_lfanew + 4] != b"PE\0\0":
        return "no PE header"
    opt = e_lfanew + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    if magic != 0x20B:  # PE32+ only (x64 / arm64)
        return "not PE32+"
    checksum_off = opt + 64
    sec_dir = opt + 112 + 4 * 8  # IMAGE_DIRECTORY_ENTRY_SECURITY
    sec_rva, sec_size = struct.unpack_from("<II", data, sec_dir)
    note = []
    if sec_size and sec_rva + sec_size == len(data):
        del data[sec_rva:]
        struct.pack_into("<II", data, sec_dir, 0, 0)
        note.append("signature removed")
    elif sec_size:
        struct.pack_into("<II", data, sec_dir, 0, 0)
        note.append("signature directory cleared")
    # PE checksum: ones'-complement sum of all 32-bit words except the checksum
    # field, folded to 16 bits, plus the file length.
    struct.pack_into("<I", data, checksum_off, 0)
    padded = bytes(data) + b"\0" * (-len(data) % 4)
    words = np.frombuffer(padded, dtype="<u4")
    total = int(words.sum(dtype=np.uint64))
    while total >> 32:
        total = (total & 0xFFFFFFFF) + (total >> 32)
    total = (total & 0xFFFF) + (total >> 16)
    total = (total + (total >> 16)) & 0xFFFF
    struct.pack_into("<I", data, checksum_off, (total + len(data)) & 0xFFFFFFFF)
    note.append("checksum recomputed")
    path.write_bytes(bytes(data))
    return ", ".join(note)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vendor(wheel: Path, dest: Path) -> Path:
    dest = dest.resolve()
    target = dest / "onnxruntime"
    if target.exists():
        shutil.rmtree(target)
    licenses_dir = dest / "licenses" / "onnxruntime"
    if licenses_dir.exists():
        shutil.rmtree(licenses_dir)
    licenses_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            bad = [n for n in names if n.startswith("/") or ".." in Path(n).parts]
            if bad:
                raise SystemExit(f"refusing to extract suspicious paths: {bad[:3]}")
            zf.extractall(tmp)
        src = Path(tmp) / "onnxruntime"
        if not src.is_dir():
            raise SystemExit("wheel does not contain an onnxruntime/ package")
        dist_info = next(Path(tmp).glob("onnxruntime-*.dist-info"), None)

        def ignore(dirpath, entries):
            rel = Path(dirpath).relative_to(src)
            skip = set()
            for e in entries:
                p = Path(dirpath) / e
                if p.is_dir() and (e in DROP_DIRS):
                    skip.add(e)
                elif p.is_file() and (e in DROP_FILES or e.endswith((".pyc", ".pyi"))):
                    skip.add(e)
                elif p.is_file() and e.startswith(DROP_BINARY_PREFIXES):
                    skip.add(e)
                elif p.is_file() and rel == Path(".") and e not in {"__init__.py", "LICENSE", "ThirdPartyNotices.txt", "Privacy.md"}:
                    skip.add(e)
            return skip

        shutil.copytree(src, target, ignore=ignore)
        for name in ("LICENSE", "ThirdPartyNotices.txt", "Privacy.md"):
            p = target / name
            if p.exists():
                shutil.copy2(p, licenses_dir / name)
        if dist_info is not None and (dist_info / "METADATA").exists():
            shutil.copy2(dist_info / "METADATA", licenses_dir / "METADATA")

    # Give the extension module a private pybind11 internals id.
    patched = []
    for ext in (target / "capi").iterdir():
        if ext.suffix in {".pyd", ".so", ".dylib"} or ".so." in ext.name:
            data = ext.read_bytes()
            count = data.count(PYBIND_ID_OLD)
            if count:
                ext.write_bytes(data.replace(PYBIND_ID_OLD, PYBIND_ID_NEW))
                extra = ""
                if ext.suffix == ".pyd":
                    extra = "; " + _pe_fixup(ext)
                patched.append(f"{ext.name} ({count} occurrence{'s' if count > 1 else ''}{extra})")
    if not patched:
        raise SystemExit("pybind11 internals id not found in any extension module; refusing to continue")

    # Rewrite imports in every Python file.
    changed = 0
    for py in target.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        new = rewrite_imports(text)
        if new != text:
            py.write_text(new, encoding="utf-8")
            changed += 1
    leftovers = []
    for py in target.rglob("*.py"):
        for line in py.read_text(encoding="utf-8").splitlines():
            if re.match(r"^\s*(from|import)\s+onnxruntime\b", line):
                leftovers.append(f"{py}: {line.strip()}")
    if leftovers:
        raise SystemExit("unrewritten imports remain:\n" + "\n".join(leftovers))

    version = re.search(r'__version__\s*=\s*"([^"]+)"', (target / "__init__.py").read_text()).group(1)
    (target / "VENDORED.txt").write_text(
        "This directory is a trimmed, import-rewritten copy of the onnxruntime Python package.\n"
        f"version:      {version}\n"
        f"source wheel: {wheel.name}\n"
        f"sha256:       {sha256(wheel)}\n"
        f"namespace:    {VENDOR_NAMESPACE}\n"
        "license:      MIT (see LICENSE); third-party notices in ThirdPartyNotices.txt\n"
        f"dropped:      {', '.join(sorted(DROP_DIRS - {'__pycache__'}))}; {', '.join(sorted(DROP_FILES))}; "
        f"{', '.join(DROP_BINARY_PREFIXES)}* (extension is statically linked)\n"
        f"patched:      pybind11 internals id {PYBIND_ID_OLD.decode()}* -> {PYBIND_ID_NEW.decode()}* "
        "so this copy can coexist with a user-installed onnxruntime; on Windows the\n"
        "              Authenticode signature of the .pyd is removed and its PE checksum recomputed\n",
        encoding="utf-8",
    )
    binaries = sorted(p.name for p in (target / "capi").iterdir() if p.suffix in {".pyd", ".dll", ".so", ".dylib"} or ".so." in p.name)
    print(f"vendored onnxruntime {version} from {wheel.name} -> {target}")
    print(f"  rewrote imports in {changed} files; binaries: {binaries}")
    print(f"  private pybind11 internals id in: {', '.join(patched)}")
    return target


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wheel", required=True, type=Path, help="onnxruntime wheel for the target platform")
    ap.add_argument("--dest", default=Path("src/arabic_ocr/_vendor"), type=Path)
    args = ap.parse_args(argv)
    vendor(args.wheel, args.dest)


if __name__ == "__main__":
    main()
