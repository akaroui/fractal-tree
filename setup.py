from setuptools import setup

cython_compile = False
if cython_compile:
    from build_geodesic import setup_geo

    setup_geo()

setup(
    name="FractalTree",
    version="0.0.44",
    url="https://github.com/GaetanDesrues/fractal-tree",
    packages=["FractalTree"],
    include_package_data=True,
    package_data={
        "FractalTree/core": ["cla.h", "cla.pxd", "rect.pyx"],
    },
)
