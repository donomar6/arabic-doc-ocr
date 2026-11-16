"""Build hooks for the platform-specific wheel.

All metadata lives in pyproject.toml. This file only (1) marks the
distribution as binary (it carries the vendored onnxruntime extension) and
(2) lets ``tools/build_wheel.py`` set the wheel tag for the *target*
interpreter/platform through environment variables, so a Windows cp310 wheel
can be assembled on any build host:

    ARABIC_OCR_PY_TAG=cp310 ARABIC_OCR_ABI_TAG=cp310 ARABIC_OCR_PLAT_TAG=win_amd64
"""
import os

from setuptools import setup
from setuptools.dist import Distribution

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        py, abi, plat = super().get_tag()
        py = os.environ.get("ARABIC_OCR_PY_TAG", py)
        abi = os.environ.get("ARABIC_OCR_ABI_TAG", abi)
        plat = os.environ.get("ARABIC_OCR_PLAT_TAG", plat)
        return py, abi, plat


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(cmdclass={"bdist_wheel": bdist_wheel}, distclass=BinaryDistribution)
