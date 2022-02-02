from setuptools import setup

cython_compile = False
if cython_compile:
    from build_geodesic import setup_geo

    setup_geo()

setup(
    name="FractalTree",
    version="0.0.42",
    url="https://github.com/GaetanDesrues/fractal-tree",
    packages=["FractalTree"],
)
