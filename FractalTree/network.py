import logging
from functools import partial
from multiprocessing import Pool

import numpy as np
import pyvista
import treefiles as tf
from MeshObject import Mesh
from tqdm import tqdm


class PurkinjeNetwork(Mesh):
    """
    Class intended to be used after the purkinje is created, i.e. loading the file
    created by `FractalTree.generator.PurkinjeGenerator.export_to_vtk`
    """

    def compute_distance(self, name="distances", optimized=True):
        """
        Main method to compute geodesic distances on network
        :param name:
        :param optimized:
        """
        # # conductivity
        # idx = 134
        # pap = m.pointIdsAroundPoint
        # keep = [idx]
        # for _ in range(3):
        #     x = []
        #     for i in np.unique(keep):
        #         x.extend(pap[i])
        #     keep.extend(x)
        # cond = np.ones(m.nbPoints)
        # cond[keep] = 0
        cond = None

        # d = self.python_distances()
        d = self.compute_distances(cond=cond, optimized=optimized)
        self.addPointData(d, name)
        if self.filename:
            self.write(self.filename)
        return d

    @tf.timer
    def compute_geodesic(self, name="distances", b=0, cond=None, optimized=True):
        """
        C++ + prange parallelism. Best option so far
        Compute for each point the shortest path from node b=0
        """
        from FractalTree.core.rect import geodesic_distance

        n = self.nbPoints
        pap = self.pointIdsAroundPoint
        c = tf.none(cond, np.ones(n))
        to_compute = self.end_nodes if optimized else range(n)

        log.info(f"Starting to compute geodesic distance")
        d = geodesic_distance(idx=to_compute, b=b, n=n, pts=self.pts, pap=pap, cond=c)

        self.addPointData(d, "distances")
        if self.filename:
            self.write(self.filename)
        return d

    @tf.timer
    def compute_distances(self, idx=0, cond=None, optimized=True):
        """
        Compute for each point the shortest path from node 0

        ,cpu,pts,time_s
        default,7706,19.1
        default,8304,21.3
        default,9714,28.0
        1/2,8898,25.3
        1,9523,26.9
        2,9626,32.2
        3,7894,20.0
        4,9861,33.6
        """
        from geodesic_ import compute_distance

        n = self.nbPoints
        pap = self.pointIdsAroundPoint
        shape = (n, len(max(pap, key=lambda x: len(x))))
        pts_a_pts = np.ones(shape, dtype=int) * -1
        for i, x in enumerate(pap):
            pts_a_pts[i][: len(x)] = x

        c = tf.none(cond, np.ones(n))
        geo = partial(compute_distance, b=idx, pts=self.pts, pap=pts_a_pts, cond=c, n=n)

        if optimized:
            to_compute = self.end_nodes
            m = len(to_compute)
        else:
            to_compute = range(n)
            m = n

        with Pool() as pool:
            n_dist = np.array(list(tqdm(pool.imap(geo, to_compute), total=m)))

        _dist = np.zeros(n)
        _dist[to_compute] = n_dist
        return _dist

    @tf.timer
    def python_distances(self):
        """
        Python version of `PurkinjeNetwork.compute_distances`

        Ex (compute purk network distances on 1449 pts):
        sequential: 28.7s
        parallel: 8.3s
        parallel (cython): 2.5s
        """
        n = self.nbPoints

        # Parallel
        geo = partial(
            python_geodesic, b=0, pts=self.pts, pap=self.pointIdsAroundPoint, n=n
        )
        with Pool() as pool:
            _dist = np.array(list(tqdm(pool.imap(geo, range(n)), total=n)))

        # # Sequential
        # _dist = np.zeros(n)
        # for i in tqdm(range(self.nbPoints)):
        #     _dist[i] = python_geodesic(
        #         i, 0, pts=self.pts, pap=self.pointIdsAroundPoint, n=n
        #     )

        return _dist

    @property
    def end_nodes(self):
        pcon = self.pointIdsAroundPoint
        endnodes = []
        for i, x in enumerate(pcon):
            if len(x) != 2 and i != 0:
                endnodes.append(i)
        return endnodes

    def end_nodes_mesh(self, endnodes=None):
        endnodes = tf.none(endnodes, self.end_nodes)
        m = PurkinjeNetwork.from_elements(self.pts[endnodes])
        for x in self.pointDataNames:
            m.addPointData(self.getPointDataArray(x)[endnodes], x)
        return m

    def prune_close_points(self, th):
        """
        Remove one point from all couples of endnodes with euclidian distance < th
        """
        endnodes = self.end_nodes
        n = len(endnodes)
        pts = self.pts[endnodes]

        to_rm, to_keep = [], []
        for i in range(n - 1):
            for j in range(i + 1, n):
                # if j not in to_rm:
                if np.linalg.norm(pts[j] - pts[i]) < th:
                    to_rm.append(j)
                    # to_keep.append(i)

        # to_rm = [x for x in to_rm if x not in np.unique(to_keep)]
        return np.unique(to_rm)
        # return np.unique(to_keep)

    @property
    def pv(self):
        return self.as_pyvista()

    def as_pyvista(self) -> pyvista.PolyData:
        line = pyvista.PolyData()
        line.points = self.pts
        line.lines = np.column_stack(([2] * self.nbCells, self.cells.as_np()))
        for x in self.pointDataNames:
            line[x] = self.getPointDataArray(x)
        return line

    def remove(self, to_rm, inplace=True):
        """
        Return new purkinje mesh, without points identified by `to_rm` ids
        """
        keep_mask = np.ones(self.nbPoints, dtype=bool)
        keep_mask[to_rm] = False
        pts = self.pts[keep_mask]

        cor, n = [], -1
        for i, x in enumerate(keep_mask):
            if x:
                n += 1
                cor.append(n)
            else:
                cor.append(-1)

        cells = np.array(
            [
                (cor[c.ids[0]], cor[c.ids[1]])
                for c in self.cells
                if keep_mask[c.ids[0]] and keep_mask[c.ids[1]]
            ]
        )

        mp = PurkinjeNetwork.from_elements(pts, cells)
        mp.clean()  # remove orpheans nodes
        for x in self.pointDataNames:
            mp.addPointData(self.getPointDataArray(x)[keep_mask], x)

        if inplace:
            self.data = mp.data
            # self._pointIdsAroundPoint = None
        return mp


def python_geodesic(a, b, pts, pap, n) -> float:
    """
    Python version of the cython version `FractalTree.geodesic_.compute_distance`
    """
    if a == b:
        return 0

    endnodes = []
    visited = np.zeros(n, dtype=bool)
    d = np.zeros(n)
    visited[0] = True
    visited[a] = True
    origins = [a]
    while len(origins) > 0:
        dmin = np.min(endnodes) if len(endnodes) > 0 else 1e10
        new_origins = []
        for y in origins:
            around = pap[y]
            for x in around:
                dcur = dist(d[y], pts[y], pts[x])
                if x == b:
                    endnodes.append(dcur)
                elif not visited[x]:
                    if dcur < dmin:
                        new_origins.append(x)
                    d[x] = dcur
                    visited[x] = True
                else:
                    _d = dcur
                    d[x] = min(_d, d[x])

            origins = new_origins

    if len(endnodes) == 0:
        return -1
    return np.min(endnodes)


def dist(dini, a, g):
    return dini + np.linalg.norm(g - a)


# class PurkinjeNetwork:
#     """
#     Class intended to be used after the purkinje is created, i.e. loading the file
#     created by `FractalTree.generator.PurkinjeGenerator.export_to_vtk`
#     """
#
#     def compute_distance(self, name="distances", optimized=True):
#         m = self.mesh
#
#         # # conductivity
#         # idx = 134
#         # pap = m.pointIdsAroundPoint
#         # keep = [idx]
#         # for _ in range(3):
#         #     x = []
#         #     for i in np.unique(keep):
#         #         x.extend(pap[i])
#         #     keep.extend(x)
#         # cond = np.ones(m.nbPoints)
#         # cond[keep] = 0
#         cond = None
#
#         # d = m.python_distances()
#         d = m.compute_distances(cond=cond, optimized=optimized)
#         m.addPointData(d, name)
#         if m.filename:
#             m.write(m.filename)
#
#     def __init__(self, src: TMeshLoadable):
#         self.mesh = PurkinjeNetwork.load(src)
#         # assert self.mesh.hasPointData("distances")
#         # assert self.mesh.hasPointData("fascicles")
#
#         self._pvmesh = None
#         self.couples = None
#
#     # def from_plane(self, ratio):
#     #     to_rm = self.get_above_plane_indices(ratio)
#     #     mp = self.remove(to_rm)
#     #     return type(self)(mp)
#
#     # def get_above_plane_indices(self, ratio):
#     #     m = self.mesh
#     #     z, a, b = m.pts[:, 2], m.bounds[4], m.bounds[5]
#     #     distances = m.getPointDataArray("distances")
#     #     idx_z = np.squeeze(np.where(z > a + ratio * (b - a)))
#     #     above_plane = []
#     #     dth = np.quantile(distances, 0.2)
#     #     for x in idx_z:
#     #         if distances[x] > dth:  # his bundle
#     #             above_plane.extend(self.branches.after(x))
#     #     return np.unique(above_plane)
#
#     # @property
#     # def branches(self):
#     #     if self.couples is None:
#     #         self.add_direction_vectors()
#     #
#     #     next_vertices = defaultdict(list)
#     #     for a, b in self.couples:
#     #         next_vertices[b].append(a)
#     #
#     #     return Branches(next_vertices)
#
#     # @property
#     # def pv(self):
#     #     if self._pvmesh is None:
#     #         self._pvmesh = self.as_pyvista()
#     #     return self._pvmesh
#     #
#     # def as_pyvista(self) -> pyvista.PolyData:
#     #     line = pyvista.PolyData()
#     #     line.points = self.mesh.pts
#     #     line.lines = np.column_stack(([2] * self.mesh.nbCells, self.mesh.cells.as_np()))
#     #     for x in self.mesh.pointDataNames:
#     #         line[x] = self.mesh.getPointDataArray(x)
#     #     return line
#
#     # def add_direction_vectors(self, name: str = "vectors"):
#     #     m = self.mesh
#     #     pts = m.pts
#     #
#     #     ff = np.zeros((m.nbPoints, 3))
#     #     couples = []
#     #     times = m.getPointDataArray("distances")
#     #
#     #     for c in m.cells:
#     #         assert c.type is Mesh.LINE
#     #         a, b = c.ids[0], c.ids[1]
#     #         a, b = (b, a) if times[a] < times[b] else (a, b)
#     #         ff[a] = pts[b] - pts[a]
#     #         couples.append((a, b))
#     #
#     #     self.pv[name] = ff
#     #     self.pv.set_active_vectors(name, preference="point")
#     #     self.couples = couples
#
#     def remove(self, to_rm, inplace=True):
#         """
#         Return new purkinje mesh, without points identified by `to_rm` ids
#         """
#         m = self.mesh
#         keep_mask = np.ones(m.nbPoints, dtype=bool)
#         keep_mask[to_rm] = False
#         pts = m.pts[keep_mask]
#
#         cor, n = [], -1
#         for i, x in enumerate(keep_mask):
#             if x:
#                 n += 1
#                 cor.append(n)
#             else:
#                 cor.append(-1)
#
#         cells = np.array(
#             [
#                 (cor[c.ids[0]], cor[c.ids[1]])
#                 for c in m.cells
#                 if keep_mask[c.ids[0]] and keep_mask[c.ids[1]]
#             ]
#         )
#
#         mp = PurkinjeNetwork.from_elements(pts, cells)
#         for x in m.pointDataNames:
#             mp.addPointData(m.getPointDataArray(x)[keep_mask], x)
#
#         if inplace:
#             self.mesh = mp
#         return mp


# class Branches(dict):
#     def _stop(self, l):
#         for x in l:
#             if x in self:
#                 return True
#         return False
#
#     def after(self, idx):
#         if idx not in self:
#             return []
#
#         ori = self[idx]
#         arr = [idx] + ori
#         while self._stop(ori):
#             ori = [y for x in ori if x in self for y in self[x]]
#             arr.extend(ori)
#
#         return arr


log = logging.getLogger(__name__)
