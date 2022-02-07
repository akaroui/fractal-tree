import logging
import sys
from random import shuffle

import numpy as np
import treefiles as tf
from MeshObject import Object, TObjectLoadable
from tqdm import tqdm

from FractalTree.Branch3D import set_log_level, Nodes, Branch
from FractalTree.Mesh import Mesh


class Curvature:
    def __init__(self, src: TObjectLoadable, fractal_mesh, cpt: bool):
        self.active = cpt
        self.m = Object.load(src, type="obj")
        self.m.convertToPolyData()
        self.m.filterCellsTo(Object.TRIANGLE)
        self.m.clean()
        self.fm = fractal_mesh

        self.curv = None
        self.n_segments = None
        self.cells = self.m.cells.as_np()
        self.pts = self.m.pts

        # self.bench_curv()

    def remesh(self):
        sol = 1 - self.curv
        sol = tf.beta_(3, 1, sol)  # filtered curvature
        lmin, lmax = 0.5, 3  # mm
        sol = (1 - sol) * lmin + sol * lmax
        # self.m.addPointData(sol, "curvature")
        self.m.remesh(sol=sol, mmg_opt=self.m.mmg_options())
        self.m.convertToPolyData()
        self.m.smooth()
        # self.m.plot(show_edges=True)  # , scalars="curvature")
        self.m.write(type="obj")

    def compute_curv(self):
        cu = self.m.compute_curvature("curvature", method='mean')
        cu = np.abs(cu)
        cu = (cu - cu.mean()) / (cu.std())
        cu = (cu - cu.min()) / np.max(cu - cu.min())

        pi = self.m.pointIdsAroundPoint
        k = 5
        for _ in range(k):
            cu = np.array([np.quantile(cu[pi[i]], 0.8) for i in range(self.m.nbPoints)])
        cu = (cu - cu.min()) / np.max(cu - cu.min())

        self.curv = cu

        cu = tf.beta_(1, 5, cu)  # filtered curvature
        nmin, nmax = 3, 15
        cu = (nmax - nmin) * cu + nmin  # mapped to n segments
        self.n_segments = cu

        self.m.addPointData(self.n_segments, "n_segments")
        # self.m.plot(scalars="n_segments")

    def bench_curv(self):
        cu = self.curv
        print("min:", cu.min())
        print("max:", cu.max())
        print("mean:", cu.mean())
        print("std:", cu.std())
        breakpoint()

    def get_nsegments(self, length, l_segment, coord):
        coord, tri = self.fm.project_new_point(coord)
        if not self.active or tri == -1:
            return int(length / l_segment)
        else:
            ids = self.cells[tri]
            # pts = self.pts[ids]  # TODO: barycentric coordinates interpolation
            return int(np.mean(self.n_segments[ids]))


def Fractal_Tree_3D(param, use_curvature=False, log_level=logging.INFO):
    """This fuction creates the fractal tree.
    Args:
        param (Parameters object): this object contains all the parameters that define the tree. See the parameters module documentation for details:
        log_level
        use_curvature

    Returns:
        branches (dict): A dictionary that contains all the branches objects.
        nodes (nodes object): the object that contains all the nodes of the tree.
    """
    set_log_level(log_level)
    log.setLevel(log_level)

    # Read Mesh
    m = Mesh(param.meshfile)

    # Set up curvature
    curvature = Curvature(param.meshfile, m, cpt=use_curvature)
    if use_curvature:
        curvature.compute_curv()
        curvature.remesh()
        m = Mesh(param.meshfile)

    # Define the initial direction
    init_dir = (param.second_node - param.init_node) / np.linalg.norm(
        param.second_node - param.init_node
    )

    # Initialize the nodes object, contains the nodes and all the distance functions
    nodes = Nodes(param.init_node)
    # Project the first node to the mesh.
    init_node = nodes.nodes[0]
    init_node[2] -= 0.5  # Bug when coord is on a mesh vertex
    point, tri = m.project_new_point(init_node)
    if tri >= 0:
        init_tri = tri
    else:
        log.error("initial point not in mesh, maybe 'Bug when coord is on a mesh vertex'")
        sys.exit(0)

    # Initialize the dictionary that stores the branches objects
    branches = {}
    last_branch = 0
    # Compute the first branch
    branches[last_branch] = Branch(
        m,
        0,
        init_dir,
        init_tri,
        param.init_length,
        0.0,
        0.0,
        nodes,
        [0],
        int(param.init_length / param.l_segment),
    )
    branches_to_grow = []
    branches_to_grow.append(last_branch)

    ien = []
    for i_n in range(len(branches[last_branch].nodes) - 1):
        ien.append(
            [branches[last_branch].nodes[i_n], branches[last_branch].nodes[i_n + 1]]
        )
    # To grow fascicles
    if param.Fascicles:
        brother_nodes = []
        brother_nodes += branches[0].nodes
        for i_branch in range(len(param.fascicles_angles)):
            last_branch += 1
            angle = param.fascicles_angles[i_branch]
            branches[last_branch] = Branch(
                m,
                branches[0].nodes[-1],
                branches[0].dir,
                branches[0].tri,
                param.fascicles_length[i_branch],
                angle,
                0.0,
                nodes,
                brother_nodes,
                int(param.fascicles_length[i_branch] / param.l_segment),
            )
            brother_nodes += branches[last_branch].nodes

            for i_n in range(len(branches[last_branch].nodes) - 1):
                ien.append(
                    [
                        branches[last_branch].nodes[i_n],
                        branches[last_branch].nodes[i_n + 1],
                    ]
                )
        branches_to_grow = list(range(1, len(param.fascicles_angles) + 1))
        fasc_nodes = brother_nodes
        # print(fasc_nodes)

    # breakpoint()

    for _ in range(param.N_it):  # tqdm
        shuffle(branches_to_grow)
        new_branches_to_grow = []
        for g in branches_to_grow:
            angle = -param.branch_angle  # * np.random.choice([-1, 1])
            for j in range(2):
                brother_nodes = []
                brother_nodes += branches[g].nodes
                if j > 0:
                    brother_nodes += branches[last_branch].nodes

                # Add new branch
                last_branch += 1
                # print(last_branch)
                l = param.length + np.random.normal(0, param.std_length)
                if l < param.min_length:
                    l = param.min_length

                # print(branches[g].nodes[-1])
                # if len(branches[g].queue) > 0:
                #     breakpoint()

                branches[last_branch] = Branch(
                    m,
                    branches[g].nodes[-1],
                    branches[g].dir,
                    branches[g].tri,
                    l,
                    angle,
                    param.w,
                    nodes,
                    brother_nodes,
                    curvature.get_nsegments(
                        param.length,
                        param.l_segment,
                        branches[g].queue[-1],
                    ),
                    fasc_nodes=fasc_nodes,
                )
                # Add nodes to IEN
                for i_n in range(len(branches[last_branch].nodes) - 1):
                    ien.append(
                        [
                            branches[last_branch].nodes[i_n],
                            branches[last_branch].nodes[i_n + 1],
                        ]
                    )

                # Add to the new array
                if branches[last_branch].growing:
                    new_branches_to_grow.append(last_branch)

                branches[g].child[j] = last_branch
                angle = -angle
        branches_to_grow = new_branches_to_grow

    if param.save:
        xyz = np.array(nodes.nodes)

        if param.save_paraview:
            from FractalTree.ParaviewWriter import write_line_VTU

            log.info("Finished growing, writing paraview file")
            write_line_VTU(xyz, ien, param.filename + ".vtu")

        np.savetxt(param.filename + "_ien.txt", ien, fmt="%d")
        np.savetxt(param.filename + "_xyz.txt", xyz)
        np.savetxt(param.filename + "_endnodes.txt", nodes.end_nodes, fmt="%d")

    return branches, nodes


log = logging.getLogger(__name__)
