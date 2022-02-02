def install():
    """Deal with compilation after install with pip"""
    import sys
    import subprocess
    import treefiles as tf

    subprocess.call(
        [sys.executable, tf.f(__file__) / "../build_geodesic.py", "build_ext", "--inplace"]
    )
