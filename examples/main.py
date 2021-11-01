import logging

import numpy as np
import pyvista
import treefiles as tf
from MeshObject import Mesh

from FractalTree.generator import PurkinjeGenerator
from FractalTree.network import PurkinjeNetwork


@tf.timer
def generate(root):
    gen = PurkinjeGenerator(root)
    m = Mesh.load(root.surf)
    gen.generate_purkinje(
        "lv",
        m,
        pt_start=np.array(m.closestPoint([0, 2, 0])[1]),
        pt_direc=np.array(m.closestPoint([0, 2, -1])[1]),
        cfg={
            "init_length_lv": 1,
            "fascicles_angles_lv": [0],
            "fascicles_length_lv": [0.7],
            "length_lv": 0.3,
            "min_length_lv": 0.01,
            "l_segment_lv": 0.1,
            "N_it_lv": 40,
            "w_lv": 0.2,
        },
        use_curvature=True,
    )


def pp(root):
    purk = PurkinjeNetwork(root.purk)

    purk.compute_distance("distances")
    # purk.remove([6,7,8])
    # purk = purk.from_plane(0.75)

    with tf.PvPlot() as p:
        p.add_mesh(
            purk.as_pyvista().tube(radius=0.01),
            # scalars="distances",  # fascicles
            # show_scalar_bar=False,
            # show_edges=True,
            opacity=0.9,
        )
        # p.add_mesh(purk.mesh, show_edges=True, opacity=0.5)
        # p.add_mesh(
        #     pyvista.PolyData(purk.mesh.point(500)),
        #     render_points_as_spheres=True,
        #     point_size=40,
        #     color="red",
        # )
        p.add_mesh(Mesh.load(root.surf), show_edges=True, opacity=0.5)
        m1 = pyvista.PolyData(purk.end_nodes_mesh.pts)
        m1['distances'] = purk.end_nodes_mesh.getPointDataArray('distances')
        p.add_mesh(
            m1,
            render_points_as_spheres=True,
            point_size=20,
            # scalars='distances',
            # color='red'
        )


def create_surf(fname):
    m = Mesh.Sphere(radius=1, theta_res=50, phi_res=50)
    m.dilate((1, 0.4, 1))
    m.addPointData(m.pts[:, 2], "z")
    m.threshold((-1, 0), "z")
    m.triangulate(m.mmg_options(hmax=0.1, hmin=0.1, hgrad=1, nr=True))
    # m.triangulate(m.mmg_options(hmax=0.75, hmin=0.75, hgrad=1, nr=True))
    # sol = m.pts[:, 2]
    # sol = (sol - sol.min()) / np.max(sol - sol.min())
    # sol = tf.beta_(5, 2).cdf(sol)
    # m.remesh(25e-3 + sol * 25e-1, mmg_opt=m.mmg_options(nr=True))
    m.write(fname)

    # with tf.PvPlot(title="my ventricle") as p:
    #     p.add_mesh(m.pv, show_edges=True)


log = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log = tf.get_logger()

    _root = tf.Tree.from_file(__file__, "root").dump()

    # create_surf(_root.surf)
    # generate(_root)
    pp(_root)
