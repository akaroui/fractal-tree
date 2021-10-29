import logging
from typing import Union

import numpy as np
import treefiles as tf
from MeshObject import Mesh, TMeshLoadable

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
        self.mesh = None

    def generate_purkinje(
        self,
        vent: str,
        endo: TMeshLoadable,
        pt_start,
        pt_direc,
        cfg,
    ):
        """
        High level method to generate the purkinje and onset points, and export them
        """

        def c(x):
            """
            Return the parameter `x` related to `vent` purkinje generation
            """
            return cfg[f"{x}_{vent}"]

        self.select_ventricle(
            vent=vent, endo=endo, pt_start=pt_start, pt_direc=pt_direc
        )
        self.get_params(c)
        self.generate()
        self.export_to_vtk()
        self.compute_distances()
        self.compute_fascicles()
        self.export_end_nodes()
        self.write()

    def select_ventricle(self, vent: str, endo: TMeshLoadable, pt_start, pt_direc):
        self.vent = vent.lower()
        assert self.vent in ["lv", "rv"]
        self.root.file(
            s=f"endo_{self.vent}.obj",
            f=f"purkinje_{self.vent}",
            voronoi=f"voronoi_{self.vent}.vtk",
        )

        # Convert endo to .obj
        self.f_endo = endo
        Mesh.load(self.f_endo).write(self.root.s, type="obj")

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
        param.filename = self.root.f
        param.save_paraview = False

        param.init_node = self.pt_start
        param.second_node = self.pt_direc

        param.init_length = cfg("his_length")
        param.length = cfg("length")
        param.l_segment = cfg("l_segment")
        param.N_it = cfg("N_it")
        param.fascicles_angles = cfg("fascicles_angles")
        param.fascicles_length = cfg("fascicles_length")
        param.branch_angle = cfg("branch_angle")

        self.params = param
        return param

    def generate(self, params: Parameters = None):
        if params is None:
            params = self.params
        self.branches, self.nodes = Fractal_Tree_3D(params)

    def export_to_vtk(self):
        self.check_result(mesh=False)

        lines = []
        for b in self.branches.values():
            for i in range(1, len(b.nodes)):
                lines.append((b.nodes[i - 1], b.nodes[i]))

        xyz = np.array(self.nodes.nodes)
        self.mesh = Mesh.from_elements(xyz, np.array(lines))
        # self.mesh.write(self.root.path("test.vtk"))

    def compute_distances(self):
        self.check_result()

        xyz = np.array(self.nodes.nodes)
        d = np.zeros(xyz.shape[0])

        def dist(dini, a, g):
            return dini + np.linalg.norm(g - a)

        for b in self.branches.values():
            idx = b.nodes
            for i in range(1, len(idx)):
                d[idx[i]] = dist(d[idx[i - 1]], xyz[idx[i - 1]], xyz[idx[i]])

        self.mesh.addPointData(d, "distances")
        return d

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

    def add_end_nodes(self):
        endpoints_idx = np.loadtxt(self.root.f + "_endnodes.txt", dtype=int)
        d = np.zeros(self.mesh.nbPoints)
        d[endpoints_idx] = 1
        self.mesh.addPointData(d, "endnodes")

    def export_end_nodes(self):
        """
        Export all end points to vtk
        """
        endpoints_idx = np.loadtxt(self.root.f + "_endnodes.txt", dtype=int)
        xyz = np.array(self.nodes.nodes)[endpoints_idx]
        pa = {
            k: self.mesh.getPointDataArray(k)[endpoints_idx]
            for k in self.mesh.pointDataNames
        }
        if "endnodes" in pa:
            pa.pop("endnodes")

        m = Mesh.from_elements(xyz, point_arrays=pa)
        fname = self.root.f + "_endp.vtk"
        m.write(fname)
        log.debug(f"Wrote end points to file://{fname}")

    def write(self, fname: str = None, **kwargs):
        if fname is None:
            fname = self.root.f + ".vtk"
        self.mesh.write(fname, **kwargs)


log = logging.getLogger(__name__)
