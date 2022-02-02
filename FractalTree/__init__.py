def install():
    """Deal with compilation after install with pip"""
    import sys
    import subprocess

    subprocess.call([sys.executable, "build_geodesic.py", "build_ext", "--inplace"])