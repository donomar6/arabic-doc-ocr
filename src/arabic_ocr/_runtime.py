"""Access to the vendored onnxruntime and to the bundled model files."""
from __future__ import annotations

import atexit
import contextlib
import importlib.resources as resources
import os
from pathlib import Path
from typing import Optional

_exit_stack = contextlib.ExitStack()
atexit.register(_exit_stack.close)
_model_paths = {}


def model_path(name: str) -> Path:
    """Filesystem path of a bundled model resource (extracted if needed)."""
    if name not in _model_paths:
        ref = resources.files("arabic_ocr").joinpath("models").joinpath(name)
        _model_paths[name] = Path(_exit_stack.enter_context(resources.as_file(ref)))
    return _model_paths[name]


def load_onnxruntime():
    """Import the private copy of onnxruntime shipped inside this package."""
    try:
        from arabic_ocr._vendor import onnxruntime as ort  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - depends on the wheel build
        raise ImportError(
            "arabic_ocr could not load its bundled onnxruntime. This wheel was built for a "
            "specific Python version and platform; make sure you installed the wheel matching "
            "your interpreter (on Windows the Visual C++ 2015-2022 redistributable is also required)."
        ) from e
    return ort


def create_session(path: os.PathLike, num_threads: Optional[int] = None):
    ort = load_onnxruntime()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if num_threads:
        so.intra_op_num_threads = int(num_threads)
        so.inter_op_num_threads = int(num_threads)
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
