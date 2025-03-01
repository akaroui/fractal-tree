# -*- coding: utf-8 -*-
"""
This module contains the Parameters class that is used to specify the input parameters of the tree.
"""

import numpy as np
import os


class Parameters:
    """Class to specify the parameters of the fractal tree.

    Attributes:
        meshfile (str): path and filename to obj file name.
        filename (str): name of the output files.
        init_node (numpy array): the first node of the tree.
        second_node (numpy array): this point is only used to calculate the initial direction of the tree and is not included in the tree. Please avoid selecting nodes that are connected to the init_node by a single edge in the mesh, because it causes numerical issues.
        init_length (float): length of the first branch.
        N_it (int): number of generations of branches.
        length (float): average lenght of the branches in the tree.
        std_length (float): standard deviation of the length. Set to zero to avoid random lengths.
        min_length (float): minimum length of the branches. To avoid randomly generated negative lengths.
        branch_angle (float): angle with respect to the direction of the previous branch and the new branch.
        # std_angle (float): standard deviation of the branch angle. Set to zero to avoid random lengths.
        w (float): repulsivity parameter.
        l_segment (float): length of the segments that compose one branch (approximately, because the lenght of the branch is random). It can be interpreted as the element length in a finite element mesh.
        Fascicles (bool): include one or more straigth branches with different lengths and angles from the initial branch. It is motivated by the fascicles of the left ventricle.
        fascicles_angles (list): angles with respect to the initial branches of the fascicles. Include one per fascicle to include.
        fascicles_length (list): length  of the fascicles. Include one per fascicle to include. The size must match the size of fascicles_angles.
        save (bool): save text files containing the nodes, the connectivity and end nodes of the tree.
        save_paraview (bool): save a .vtu paraview file. The tvtk module must be installed.

    """

    def __init__(self):
        self.meshfile = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "src", "sphere.obj"
        )
        self.filename = "sphere-line"
        self.init_node = np.array([-1.0, 0.0, 0.0])
        self.second_node = np.array([-0.964, 0.00, 0.266])
        self.init_length = 0.1
        # Number of iterations (generations of branches)
        self.N_it = 10
        # Median length of the branches
        self.length = 0.1
        # Standard deviation of the length
        self.std_length = 0  # np.sqrt(0.2) * self.length
        # Min length to avoid negative length
        self.min_length = self.length / 10.0
        self.branch_angle = 0.15
        # self.std_angle = 0
        self.w = 0.2
        # Length of the segments (approximately, because the lenght of the branch is random)
        self.l_segment = 0.01

        self.Fascicles = True
        ###########################################
        # Fascicles data
        ###########################################
        self.fascicles_angles = [-1.5, 0.2]  # rad
        self.fascicles_length = [0.5, 0.5]
        # Save data?
        self.save = True
        self.save_paraview = True

    def __repr__(self):
        return (
            f"meshfile: {self.meshfile}\n"
            f"filename: {self.filename}\n"
            f"init_node: {self.init_node}\n"
            f"second_node: {self.second_node}\n"
            f"init_length: {self.init_length}\n"
            f"N_it: {self.N_it}\n"
            f"length: {self.length}\n"
            f"std_length: {self.std_length}\n"
            f"min_length: {self.min_length}\n"
            f"branch_angle: {self.branch_angle}\n"
            # f"std_angle: {self.std_angle}\n"
            f"w: {self.w}\n"
            f"l_segment: {self.l_segment}\n"
            f"Fascicles: {self.Fascicles}\n"
            f"fascicles_angles: {self.fascicles_angles}\n"
            f"fascicles_length: {self.fascicles_length}\n"
            f"save: {self.save}\n"
            f"save_paraview: {self.save_paraview}"
        )
