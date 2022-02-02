

def install():
    """Deal with compilation after install with pip"""

    from Cython.Build import cythonize
    from setuptools import setup


    setup(
        ext_modules=cythonize(
            "core/rect.pyx",
            compiler_directives={"language_level": "3"},
        ),
    )
