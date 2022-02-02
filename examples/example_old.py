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
    # m.triangulate(m.mmg_options(hmax=1, hgrad=1, nr=True))
    # m.smooth()
    # m.filterCellsTo(Mesh.TRIANGLE)
    # m.convertToPolyData()
    # m.write()
    # m.plot(show_edges=True)
    # with tf.PvPlot() as p:
    #     p.add_mesh(m, show_edges=True)
    #     # p.add_mesh(pyvista.PolyData([m.point(2701)]), render_points_as_spheres=True, point_size=20, color='red')
    #     p.add_mesh(
    #         pyvista.PolyData([m.point(2594), m.point(355)]),
    #         render_points_as_spheres=True,
    #         point_size=20,
    #         color="red",
    #     )

    gen.generate_purkinje(
        "rv",
        m,
        # pt_start=np.array(m.closestPoint([0, 2, 0])[1]),
        # pt_direc=np.array(m.closestPoint([0, 2, -1])[1]),
        #
        # # LV
        # pt_start=m.point(2594),
        # pt_direc=m.point(355),
        # #
        # # Real endocardium lv
        # cfg={
        #     "init_length_lv": 10,
        #     "fascicles_angles_lv": [0],
        #     "fascicles_length_lv": [60],
        #     "length_lv": 20,  # 0.3,
        #     "min_length_lv": 5,
        #     "l_segment_lv": 5,  # 0.1,
        #     "N_it_lv": 25,
        #     "w_lv": 0.2,
        # },
        #
        # RV
        pt_start=m.point(841),
        pt_direc=m.point(1125),
        #
        # Real endocardium rv
        cfg={
            "init_length_rv": 10,
            "fascicles_angles_rv": [0.5],
            "fascicles_length_rv": [30],
            "length_rv": 10,  # 0.3,
            "min_length_rv": 2,
            "l_segment_rv": 2,  # 0.1,
            "N_it_rv": 12,
            "w_rv": 0.2,
        },
        #
        # Test on sphere
        # cfg={
        #     "init_length_lv": 1,
        #     "fascicles_angles_lv": [0],
        #     "fascicles_length_lv": [0.7],
        #     "length_lv": 0.5,  # 0.3,
        #     "min_length_lv": 0.01,
        #     "l_segment_lv": 0.2,  # 0.1,
        #     "N_it_lv": 20,
        #     "w_lv": 0.18,
        # },
        use_curvature=True,
    )


def pp(root):
    purk = PurkinjeNetwork.load(root.purk)
    # rmed = purk.pts[range(800, 900)]
    # purk = purk.remove(range(800, 900), inplace=False)

    # nbv = purk.mesh.prune_close_points(th=3)
    # purk.remove(nbv)
    # d = purk.compute_distance("distances", optimized=False)

    th = purk.bounds[4] + 0.8 * (purk.bounds[5] - purk.bounds[4])
    purk = purk.remove(np.where(purk.pts[:, 2] > th), inplace=False)

    d = purk.compute_geodesic(optimized=False)
    # print(d.shape)
    # breakpoint()

    mask = np.where(d == -1)
    rmed2 = purk.pts[mask]
    purk = purk.remove(mask, inplace=False)

    # purk.remove([6,7,8])
    # purk = purk.from_plane(0.75)

    with tf.PvPlot(title="Purkinje") as p:
        # Surface mesh
        p.add_mesh(Mesh.load(root.surf), show_edges=True, opacity=0.2)

        # Purkinje tubes
        p.add_mesh(
            purk.pv.tube(radius=0.35),
            # color="white",
            scalars="distances",  # fascicles
            # show_scalar_bar=False,
            # show_edges=True,
            # opacity=0.9,
        )
        # p.add_mesh(purk.mesh, show_edges=True, opacity=0.5)

        # End points
        m0 = purk.end_nodes_mesh()
        m0.write(root.endp)
        m1 = pyvista.PolyData(m0.pts)
        m1["distances"] = m0.getPointDataArray("distances")
        p.add_mesh(
            m1,
            render_points_as_spheres=True,
            point_size=20,
            # scalars='distances',
            # color='red'
        )

        # p.add_mesh(
        #     # Init nodes
        #     # pyvista.PolyData([purk.point(134)]),
        #     # to_rm
        #     rmed,  # purk.pts[range(10, 20)],
        #     render_points_as_spheres=True,
        #     point_size=25,
        #     color="red",
        #     # opacity=0.8,
        # )
        # p.add_mesh(
        #     # Init nodes
        #     # pyvista.PolyData([purk.point(134)]),
        #     # to_rm
        #     rmed2,  # purk.pts[range(10, 20)],
        #     render_points_as_spheres=True,
        #     point_size=25,
        #     color="green",
        #     # opacity=0.8,
        # )


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

    create_surf(_root.surf)
    generate(_root)
    pp(_root)
