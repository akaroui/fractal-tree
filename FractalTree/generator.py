import logging
from typing import Union

import numpy as np
import treefiles as tf
from MeshObject import Mesh, TMeshLoadable, Object, TMesh

from FractalTree.FractalTree import Fractal_Tree_3D
from FractalTree.parameters import Parameters


class PurkinjeGenerator:
    def __init__(self, out_dir: Union[str, tf.Tree]):
        self.root = out_dir
        self.f_endo = None

        self.vent = None
        self.pt_start = None
        self.pt_direc = None

        self.params = None
        self.branches = None
        self.nodes = None
        self.mesh: TMesh = None

    def generate_purkinje(
        self,
        vent: str,
        endo: TMeshLoadable,
        pt_start,
        pt_direc,
        cfg,
        use_curvature=False,
    ):
        """
        High level method to generate the purkinje and onset points, and export them
        """

        def c(x, default):
            """
            Return the parameter `x` related to `vent` purkinje generation
            """
            return cfg.get(f"{x}_{vent}", default)

        self.select_ventricle(
            vent=vent, endo=endo, pt_start=pt_start, pt_direc=pt_direc
        )
        self.get_params(c)
        self.generate(use_curvature=use_curvature)
        self.export_to_vtk()
        # self.compute_distances_2()
        # # self.compute_distances()
        # self.compute_fascicles()
        # self.export_end_nodes()
        self.write()

    def select_ventricle(self, vent: str, endo: TMeshLoadable, pt_start, pt_direc):
        self.vent = vent.lower()
        assert self.vent in ["lv", "rv"]
        self.root.file(
            surf=f"endo_{self.vent}.vtk",
            s=f"endo_{self.vent}.obj",
            purk=f"purkinje_{self.vent}.vtk",
            gen=f"generated_{self.vent}",
            # voronoi=f"voronoi_{self.vent}.vtk",
        )

        # Convert endo to .obj
        self.f_endo = endo
        m = Mesh.load(self.f_endo)
        m.write(self.root.surf)
        m.convertToPolyData().write(self.root.s, type="obj")

        self.pt_start = pt_start
        self.pt_direc = pt_direc

    def check_result(self, mesh=True):
        if self.vent is None:
            log.error(f"You must run select_ventricle after class instanciation")
        assert self.vent is not None
        assert self.branches is not None
        assert self.nodes is not None
        assert self.params is not None
        if mesh:
            assert self.mesh is not None

    def get_params(self, cfg):
        param = Parameters()
        param.meshfile = self.root.s
        param.filename = self.root.gen
        param.save_paraview = False

        param.init_node = self.pt_start
        param.second_node = self.pt_direc

        param.init_length = cfg("his_length", param.init_length)
        param.length = cfg("length", param.length)
        param.l_segment = cfg("l_segment", param.l_segment)
        param.N_it = cfg("N_it", param.N_it)
        param.fascicles_angles = cfg("fascicles_angles", param.fascicles_angles)
        param.fascicles_length = cfg("fascicles_length", param.fascicles_length)
        param.branch_angle = cfg("branch_angle", param.branch_angle)
        param.w = cfg("w", param.w)
        param.min_length = cfg("min_length", param.min_length)

        self.params = param
        return param

    def generate(self, use_curvature=False):
        self.branches, self.nodes = Fractal_Tree_3D(
            self.params, use_curvature=use_curvature
        )

    def export_to_vtk(self):
        self.check_result(mesh=False)

        lines = []
        for b in self.branches.values():
            for i in range(1, len(b.nodes)):
                lines.append((b.nodes[i - 1], b.nodes[i]))

        xyz = np.array(self.nodes.nodes)
        self.mesh = Mesh.from_elements(xyz, np.array(lines))
        # self.mesh.write(self.root.path("test.vtk"))

    def write(self, **kwargs):
        self.mesh.write(self.root.purk, **kwargs)

    # def compute_distances(self):
    #     self.check_result()
    #
    #     xyz = np.array(self.nodes.nodes)
    #     d = np.ones(xyz.shape[0]) * 1e10
    #     d[0] = 0
    #
    #     def dist(dini, a, g):
    #         return dini + np.linalg.norm(g - a)
    #
    #     for b in self.branches.values():
    #         idx = b.nodes
    #         for i in range(1, len(idx)):
    #             d[idx[i]] = min(
    #                 dist(d[idx[i - 1]], xyz[idx[i - 1]], xyz[idx[i]]), d[idx[i]]
    #             )
    #
    #     self.mesh.addPointData(d, "distances")
    #     return d
    #
    # def compute_distances_2(self):
    #     self.check_result()
    #
    #     pts = np.array(self.nodes.nodes)
    #     visited = np.zeros(pts.shape[0], dtype=bool)
    #     visited[0] = True
    #     to_visit = [1]
    #     dist = np.zeros(pts.shape[0])
    #     pt_a_pt = self.mesh.pointIdsAroundPoint
    #
    #     def get_dist(ids_around):
    #         v = visited[ids_around]
    #         nb_visited = np.count_nonzero(v)
    #
    #         if nb_visited == 0 or nb_visited > 2:
    #             return
    #         elif nb_visited == 1:
    #             ids_visited = np.array(ids_around)[v]
    #             ids_next = np.array(ids_around)[~v]
    #             return ids_visited, ids_next
    #         elif nb_visited == 2:
    #             ids_visited = np.array(ids_around)[v]
    #             shor_i = np.argmin((dist[ids_visited[0]], dist[ids_visited[1]]))
    #             ids_visited = [shor_i]
    #             ids_next = np.array(ids_around)[~v]
    #             np.append(ids_next, [int(1 - shor_i)], 0)
    #             return ids_visited, ids_next
    #         else:
    #             raise NotImplementedError
    #
    #     n = 0
    #     while not np.all(visited) and len(to_visit) > 0 and n < 500:
    #         n += 1
    #         next_visit = []
    #         for x in to_visit:
    #             # if not visited[x]:
    #             r = get_dist(pt_a_pt[x])
    #             if r:
    #                 _current = x
    #                 _visited, _next = r
    #                 prv_dist = 1e10 if not visited[x] else dist[_current]
    #                 y = dist[_visited] + np.linalg.norm(pts[_current] - pts[_visited])
    #                 if y < prv_dist:
    #                     if y[0] < 0.2:
    #                         print(y)
    #                     dist[_current] = y
    #                 next_visit.extend(_next)
    #                 visited[x] = True
    #                 # breakpoint()
    #             else:
    #                 pass
    #         to_visit = next_visit
    #     print(n)
    #     # breakpoint()
    #     self.mesh.addPointData(dist, "distances")
    #     return dist

    # xyz = np.array(self.nodes.nodes)
    # d = np.ones(xyz.shape[0]) * 1e10
    # d[0] = 0
    #
    # def dist(dini, a, g):
    #     return dini + np.linalg.norm(g - a)
    #
    # for b in self.branches.values():
    #     idx = b.nodes
    #     for i in range(1, len(idx)):
    #         d[idx[i]] = min(
    #             dist(d[idx[i - 1]], xyz[idx[i - 1]], xyz[idx[i]]), d[idx[i]]
    #         )
    #
    # self.mesh.addPointData(d, "distances")
    # return d

    def compute_fascicles(self):
        self.check_result()
        d = np.zeros(self.mesh.nbPoints)

        def register_nodes(br, key):
            for k in br.nodes:
                d[k] = key

        n_fascicle = len(self.params.fascicles_length) if self.params.Fascicles else 0
        for i in range(n_fascicle + 1):
            register_nodes(self.branches[i], i)
            # anterior fascicle: 2
            # posterior fascicle: 1

        for i in range(3, len(self.branches)):
            register_nodes(self.branches[i], d[self.branches[i].nodes[0]])

        self.mesh.addPointData(d, "fascicles")
        return d

    # def add_end_nodes(self):
    #     endpoints_idx = np.loadtxt(self.root.f + "_endnodes.txt", dtype=int)
    #     d = np.zeros(self.mesh.nbPoints)
    #     d[endpoints_idx] = 1
    #     self.mesh.addPointData(d, "endnodes")

    # def export_end_nodes(self):
    #     """
    #     Export all end points to vtk
    #     """
    #     endpoints_idx = np.loadtxt(self.root.f + "_endnodes.txt", dtype=int)
    #     xyz = np.array(self.nodes.nodes)[endpoints_idx]
    #     pa = {
    #         k: self.mesh.getPointDataArray(k)[endpoints_idx]
    #         for k in self.mesh.pointDataNames
    #     }
    #     if "endnodes" in pa:
    #         pa.pop("endnodes")
    #
    #     m = Mesh.from_elements(xyz, point_arrays=pa)
    #     fname = self.root.f + "_endp.vtk"
    #     m.write(fname)
    #     log.debug(f"Wrote end points to file://{fname}")


log = logging.getLogger(__name__)
