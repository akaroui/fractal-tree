from setuptools import setup
from Cython.Build import cythonize

setup(
    name="FractalTree",
    version="0.0.15",
    ext_modules=cythonize(
        "FractalTree/geodesic_.pyx",
        compiler_directives={"language_level": "3"},
    ),
    url="https://github.com/GaetanDesrues/fractal-tree",
    packages=["FractalTree"],
)
