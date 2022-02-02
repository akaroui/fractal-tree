from Cython.Build import cythonize
from setuptools import setup

# cython_compile = True
# if cython_compile:
#     from build_geodesic import setup_geo
#
#     setup_geo()

setup(
    name="FractalTree",
    version="0.0.45",
    url="https://github.com/GaetanDesrues/fractal-tree",
    packages=["FractalTree"],
    include_package_data=True,
    package_data={
        "FractalTree/core": ["cla.h", "cla.pxd", "rect.pyx"],
    },
    ext_modules=cythonize(
        "FractalTree/core/rect.pyx",
        compiler_directives={"language_level": "3"},
    ),
)
