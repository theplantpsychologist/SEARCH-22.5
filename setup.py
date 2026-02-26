from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "engine.math225_core",
        ["src/core/math225_core.cpp"], # This one worked
        include_dirs=["src/core"],
    ),
]

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
