

def install():
    """Deal with compilation after install with pip"""

    from Cython.Build import cythonize
    from setuptools import setup


    setup(
        ext_modules=cythonize(
            "FractalTree/core/rect.pyx",
            compiler_directives={"language_level": "3"},
        ),
    )
