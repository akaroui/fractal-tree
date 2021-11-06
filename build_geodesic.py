import sys

from setuptools import setup
from Cython.Build import cythonize

# Run `python build_geodesic.py build_ext --inplace`


def setup_geo():
    setup(
        name="FractalTree_geo",
        setup_requires=["cython", "numpy"],
        install_requires=["numpy"],
        ext_modules=cythonize(
            "FractalTree/geodesic_.pyx",
            compiler_directives={"language_level": "3"},
        ),
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(("build_ext", "--inplace"))

    setup_geo()
