import logging
from collections import defaultdict

import numpy as np
import pyvista as pv
from MeshObject import Mesh, TMeshLoadable


class PurkinjeNetwork:
    """
    Class intended to be used after the purkinje is created, i.e. loading the file
    created by `FractalTree.generator.PurkinjeGenerator.export_to_vtk`
    """

    def __init__(self, src: TMeshLoadable):
        self.mesh = Mesh.load(src)
        assert self.mesh.hasPointData("distances")
        assert self.mesh.hasPointData("fascicles")

        self._pvmesh = None
        self.couples = None

    def from_plane(self, ratio):
        to_rm = self.get_above_plane_indices(ratio)
        mp = self.remove(to_rm)
        return type(self)(mp)

    def get_above_plane_indices(self, ratio):
        m = self.mesh
        z, a, b = m.pts[:, 2], m.bounds[4], m.bounds[5]
        distances = m.getPointDataArray("distances")
        idx_z = np.squeeze(np.where(z > a + ratio * (b - a)))
        above_plane = []
        for x in idx_z:
            if distances[x] > 40:  # his bundle
                above_plane.extend(self.branches.after(x))
        return np.unique(above_plane)

    @property
    def branches(self):
        if self.couples is None:
            self.add_direction_vectors()

        next_vertices = defaultdict(list)
        for a, b in self.couples:
            next_vertices[b].append(a)

        return Branches(next_vertices)

    @property
    def pvmesh(self):
        if self._pvmesh is None:
            self._pvmesh = self.as_pyvista()
        return self._pvmesh

    @pvmesh.setter
    def pvmesh(self, obj):
        self._pvmesh = obj

    def as_pyvista(self) -> pv.PolyData:
        line = pv.PolyData()
        line.points = self.mesh.pts
        line.lines = np.column_stack(([2] * self.mesh.nbCells, self.mesh.cells.as_np()))
        for x in self.mesh.pointDataNames:
            line[x] = self.mesh.getPointDataArray(x)
        return line

    def add_direction_vectors(self, name: str = "vectors"):
        m = self.mesh
        pts = m.pts

        ff = np.zeros((m.nbPoints, 3))
        couples = []
        times = m.getPointDataArray("distances")

        for c in m.cells:
            assert c.type is Mesh.LINE
            a, b = c.ids[0], c.ids[1]
            a, b = (b, a) if times[a] < times[b] else (a, b)
            ff[a] = pts[b] - pts[a]
            couples.append((a, b))

        self.pvmesh[name] = ff
        self.pvmesh.set_active_vectors(name, preference="point")
        self.couples = couples

    def remove(self, to_rm):
        """
        Return new purkinje mesh, without points identified by `to_rm` ids
        """
        m = self.mesh
        keep_mask = np.ones(m.nbPoints, dtype=bool)
        keep_mask[to_rm] = False
        pts = m.pts[keep_mask]

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
                for c in m.cells
                if keep_mask[c.ids[0]] and keep_mask[c.ids[1]]
            ]
        )

        mp = Mesh.from_elements(pts, cells)
        for x in m.pointDataNames:
            mp.addPointData(m.getPointDataArray(x)[keep_mask], x)

        return mp

    @property
    def end_nodes(self):
        m = self.mesh
        pcon = m.pointIdsAroundPoint
        endnodes = []
        for i, x in enumerate(pcon):
            if len(x) == 1 and i != 0:
                endnodes.append(i)
        return endnodes

    @property
    def end_nodes_mesh(self):
        endnodes = self.end_nodes
        m = Mesh.from_elements(self.mesh.pts[endnodes])
        for x in self.mesh.pointDataNames:
            m.addPointData(self.mesh.getPointDataArray(x)[endnodes], x)
        return m


class Branches(dict):
    def _stop(self, l):
        for x in l:
            if x in self:
                return True
        return False

    def after(self, idx):
        if idx not in self:
            return []

        ori = self[idx]
        arr = [idx] + ori
        while self._stop(ori):
            ori = [y for x in ori if x in self for y in self[x]]
            arr.extend(ori)

        return arr


log = logging.getLogger(__name__)
