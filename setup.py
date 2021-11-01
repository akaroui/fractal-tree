from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name="FractalTree",
    version="0.0.32",
    # setup_requires=["cython", "numpy"],
    # install_requires=["numpy"],
    # ext_modules=cythonize(
    #     "FractalTree/geodesic_.pyx",
    #     compiler_directives={"language_level": "3"},
    # ),
    # ext_modules=[Extension("geodesic_", ["FractalTree/geodesic_.cpp"])],
    url="https://github.com/GaetanDesrues/fractal-tree",
    packages=["FractalTree"],
)
