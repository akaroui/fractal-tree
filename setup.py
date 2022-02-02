from setuptools import setup


setup(
    name="FractalTree",
    version="0.0.48",
    url="https://github.com/GaetanDesrues/fractal-tree",
    packages=["FractalTree"],
    include_package_data=True,
    package_data={
        "FractalTree/core": ["cla.h", "cla.pxd", "rect.pyx"],
    },
    # ext_modules=cythonize(
    #     "FractalTree/core/rect.pyx",
    #     compiler_directives={"language_level": "3"},
    # ),
)
