import sys


def install():
    """Deal with compilation after install with pip"""

    from setuptools import setup
    from Cython.Build import cythonize

    if len(sys.argv) == 1:
        sys.argv.extend(("build_ext", "--inplace"))

    # python setup.py build_ext --inplace

    setup(ext_modules=cythonize("core/rect.pyx"))
