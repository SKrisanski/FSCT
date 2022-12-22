import math
from copy import deepcopy
from multiprocessing import get_context
import networkx as nx
import numpy as np
import pandas as pd
import os
import time
from scipy import spatial  # TODO Test if sklearn kdtree is faster.
from scipy.interpolate import griddata
from skimage.measure import CircleModel, ransac
from sklearn.neighbors import NearestNeighbors
from skspatial.objects import Plane
from sklearn.neighbors import BallTree
from scripts.tools import (
    get_fsct_path,
    load_file,
    save_file,
    low_resolution_hack_mode,
    cluster_hdbscan,
    cluster_dbscan,
    get_heights_above_DTM,
    get_taper,
)
from scripts.fsct_exceptions import DataQualityError


class MeasureTree:
    def __init__(self, parameters):
        self.measure_time_start = time.time()
        self.parameters = parameters
        self.filename = self.parameters["point_cloud_filename"].replace("\\", "/")
        self.output_dir = (
            os.path.dirname(os.path.realpath(self.filename)).replace("\\", "/")
            + "/"
            + self.filename.split("/")[-1][:-4]
            + "_FSCT_output/"
        )
        self.filename = self.filename.split("/")[-1]

        self.num_cpu_cores = parameters["num_cpu_cores"]
        self.num_neighbours = parameters["num_neighbours"]
        self.slice_thickness = parameters["slice_thickness"]
        self.slice_increment = parameters["slice_increment"]

        self.plot_summary = pd.read_csv(self.output_dir + "plot_summary.csv", index_col=False)
        self.parameters["plot_radius"] = float(self.plot_summary["Plot Radius"])
        self.parameters["plot_radius_buffer"] = float(self.plot_summary["Plot Radius Buffer"])
        self.plot_area = float(self.plot_summary["Plot Area"])
        self.stem_points, headers_of_interest = load_file(
            self.output_dir + "stem_points.las",
            headers_of_interest=["x", "y", "z", "red", "green", "blue", "label", "height_above_DTM"],
        )
        self.stem_points = np.hstack((self.stem_points, np.zeros((self.stem_points.shape[0], 1))))

        print("stempoints", headers_of_interest)
        if self.parameters["low_resolution_point_cloud_hack_mode"]:
            self.stem_points = low_resolution_hack_mode(
                self.stem_points,
                self.parameters["low_resolution_point_cloud_hack_mode"],
                self.parameters["subsampling_min_spacing"],
                self.parameters["num_cpu_cores"],
            )
            save_file(self.output_dir + self.filename[:-4] + "_stem_points_hack_mode_cloud.las", self.stem_points)

        self.DTM, headers_of_interest = load_file(self.output_dir + "DTM.las")
        self.characters = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "dot",
            "m",
            "space",
            "_",
            "-",
            "semiC",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "_M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
        ]
        self.character_viz = []

        for i in self.characters:
            self.character_viz.append(
                np.genfromtxt(os.path.join(get_fsct_path("tools/numbers"), i + ".csv"), delimiter=",")
            )

        self.cyl_dict = dict(
            x=0,
            y=1,
            z=2,
            nx=3,
            ny=4,
            nz=5,
            radius=6,
            CCI=7,
            branch_id=8,
            parent_branch_id=9,
            tree_id=10,
            tree_volume=11,
            segment_angle_to_horiz=12,
            height_above_dtm=13,
        )

        self.veg_dict = dict(x=0, y=1, z=2, red=3, green=4, blue=5, label=6, height_above_dtm=7, tree_id=8)
        self.stem_dict = dict(x=0, y=1, z=2, red=3, green=4, blue=5, label=6, height_above_dtm=7, tree_id=8)
        self.tree_data_dict = dict(
            PlotId=0,
            TreeId=1,
            x_tree_base=2,
            y_tree_base=3,
            z_tree_base=4,
            DBH=5,
            CCI_at_BH=6,
            Height=7,
            Volume_1=8,
            Volume_2=9,
            Crown_mean_x=10,
            Crown_mean_y=11,
            Crown_top_x=12,
            Crown_top_y=13,
            Crown_top_z=14,
            mean_understory_height_in_5m_radius=15,
        )

        self.terrain_points, headers_of_interest = load_file(
            self.output_dir + "terrain_points.las",
            headers_of_interest=["x", "y", "z", "red", "green", "blue", "label", "height_above_DTM"],
        )
        self.terrain_points = np.hstack((self.terrain_points, np.zeros((self.terrain_points.shape[0], 1))))

        self.vegetation_points, headers_of_interest = load_file(
            self.output_dir + "vegetation_points.las",
            headers_of_interest=["x", "y", "z", "red", "green", "blue", "label", "height_above_DTM"],
        )
        self.vegetation_points = np.hstack((self.vegetation_points, np.zeros((self.vegetation_points.shape[0], 1))))

        # Remove understorey vegetation and save it.
        ground_veg_mask = (
            self.vegetation_points[:, self.veg_dict["height_above_dtm"]] <= self.parameters["ground_veg_cutoff_height"]
        )
        self.ground_veg = self.vegetation_points[ground_veg_mask]
        save_file(self.output_dir + "ground_veg.las", self.ground_veg, headers_of_interest=list(self.veg_dict))
        self.vegetation_points = self.vegetation_points[np.logical_not(ground_veg_mask)]
        veg_kdtree = spatial.cKDTree(self.vegetation_points[:, :2], leafsize=10000)

        try:
            self.cwd_points, headers_of_interest = load_file(
                self.output_dir + "cwd_points.las",
                headers_of_interest=["x", "y", "z", "red", "green", "blue", "label", "height_above_DTM"],
            )
            if self.cwd_points.shape[1] < self.stem_points.shape[1]:
                self.cwd_points = np.hstack(
                    (
                        self.cwd_points,
                        np.zeros((self.cwd_points.shape[0], self.stem_points.shape[1] - self.cwd_points.shape[1])),
                    )
                )

        except FileNotFoundError:
            self.cwd_points = np.zeros((0, self.stem_points.shape[1]))

        cwd_kdtree = spatial.cKDTree(self.cwd_points[:, :2], leafsize=10000)

        self.ground_veg_kdtree = spatial.cKDTree(self.ground_veg[:, :2], leafsize=10000)
        xmin = np.floor(np.min(self.terrain_points[:, 0]))
        ymin = np.floor(np.min(self.terrain_points[:, 1]))
        xmax = np.ceil(np.max(self.terrain_points[:, 0]))
        ymax = np.ceil(np.max(self.terrain_points[:, 1]))
        x_points = np.linspace(
            xmin, xmax, int(np.ceil((xmax - xmin) / self.parameters["vegetation_coverage_resolution"])) + 1
        )
        y_points = np.linspace(
            ymin, ymax, int(np.ceil((ymax - ymin) / self.parameters["vegetation_coverage_resolution"])) + 1
        )

        convexhull = spatial.ConvexHull(self.DTM[:, :2])
        self.ground_area = 0  # unitless. Not in m2.
        self.canopy_area = 0  # unitless. Not in m2.
        self.ground_veg_area = 0  # unitless. Not in m2.
        self.cwd_area = 0  # unitless. Not in m2.
        for x in x_points:
            for y in y_points:
                if self.inside_conv_hull(np.array([x, y]), convexhull):
                    indices = veg_kdtree.query_ball_point(
                        [x, y], r=self.parameters["vegetation_coverage_resolution"], p=10
                    )
                    ground_veg_indices = self.ground_veg_kdtree.query_ball_point(
                        [x, y], r=self.parameters["vegetation_coverage_resolution"], p=10
                    )
                    cwd_indices = cwd_kdtree.query_ball_point(
                        [x, y], r=self.parameters["vegetation_coverage_resolution"], p=10
                    )
                    self.ground_area += 1
                    if len(indices) > 5:
                        self.canopy_area += 1

                    if len(ground_veg_indices) > 5:
                        self.ground_veg_area += 1

                    if len(cwd_indices) > 5:
                        self.cwd_area += 1

        print("Canopy Cover Fraction:", self.canopy_area / self.ground_area)
        print("Understory Veg Fraction:", self.ground_veg_area / self.ground_area)
        print("Coarse Woody Debris Fraction:", self.cwd_area / self.ground_area)
        max_z = np.max(
            np.hstack(
                (
                    self.stem_points[:, 2],
                    self.vegetation_points[:, 2],
                    self.cwd_points[:, 2],
                    self.terrain_points[:, 2],
                )
            )
        )
        min_z = np.min(
            np.hstack(
                (
                    self.stem_points[:, 2],
                    self.vegetation_points[:, 2],
                    self.cwd_points[:, 2],
                    self.terrain_points[:, 2],
                )
            )
        )
        self.z_range = max_z - min_z
        self.text_point_cloud = np.zeros((0, 3))
        self.tree_measurements = np.zeros((0, 8))
        self.text_point_cloud = np.zeros((0, 3))

    def interpolate_cyl(self, cyl1, cyl2, resolution):
        """
        Convention to be used
        cyl_1 is child
        cyl_2 is parent
        """
        length = np.linalg.norm(np.array([cyl2[0], cyl2[1], cyl2[2]]) - np.array([cyl1[0], cyl1[1], cyl1[2]]))
        points_per_line = int(np.ceil(length / resolution))
        interpolated = np.zeros((0, 14))
        if cyl1.shape[0] > 0 and cyl2.shape[0] > 0:
            xyzinterp = np.linspace(cyl1[:3], cyl2[:3], points_per_line, axis=0)
            if xyzinterp.shape[0] > 0:
                interpolated = np.zeros((xyzinterp.shape[0], 14))
                interpolated[:, :3] = xyzinterp

                normal = (cyl2[:3] - cyl1[:3]) / np.linalg.norm(cyl2[:3] - cyl1[:3])

                if normal[2] < 0:
                    normal[:3] = normal[:3] * -1

                interpolated[:, 3:6] = normal

                interpolated[:, self.cyl_dict["tree_id"]] = cyl1[self.cyl_dict["tree_id"]]
                interpolated[:, self.cyl_dict["branch_id"]] = cyl1[self.cyl_dict["branch_id"]]
                interpolated[:, self.cyl_dict["parent_branch_id"]] = cyl2[self.cyl_dict["branch_id"]]
                interpolated[:, self.cyl_dict["radius"]] = np.min(
                    [cyl1[self.cyl_dict["radius"]], cyl2[self.cyl_dict["radius"]]]
                )

        return interpolated

    @classmethod
    def compute_angle(cls, normal1, normal2):
        """
        Computes the angle in degrees between two 3D vectors.

        Args:
            normal1:
            normal2:

        Returns:
            theta: angle in degrees
        """
        normal1 = np.atleast_2d(normal1)
        normal2 = np.atleast_2d(normal2)

        norm1 = normal1 / np.atleast_2d(np.linalg.norm(normal1, axis=1)).T
        norm2 = normal2 / np.atleast_2d(np.linalg.norm(normal2, axis=1)).T
        dot = np.clip(np.einsum("ij,ij->i", norm1, norm2), -1, 1)
        theta = np.degrees(np.arccos(dot))
        return theta

    def cylinder_sorting(self, cylinder_array, angle_tolerance, search_angle, distance_tolerance):
        """
        Step 1 of sorting initial cylinders into individual trees.
        For a cylinder to be joined up with another cylinder in this step, it must meet the below conditions.

        All angles are specified in degrees.
            cylinder_array:
                The Numpy array of cylinders created during cylinder fitting.

            angle_tolerance:
                Angle tolerance refers to the angle between major axis vectors of the two cylinders being queried. If
                the angle is less than "angle_tolerance", this condition is satisfied.

            search_angle:
                Search angle refers to the angle between the major axis of cylinder 1, and the vector from cylinder 1's
                centre point to cylinder 2's centre point.

            distance_tolerance:
                Cylinder centre points must be within this distance to meet this condition. Think of a ball of radius
                "distance_tolerance".

        Returns: The sorted cylinder array.
        """

        def within_angle_tolerance(normal1, normal2, angle_tolerance):
            """Checks if normal1 and normal2 are within "angle_tolerance"
            of each other."""
            theta = self.compute_angle(normal1, normal2)
            return abs((theta > 90) * 180 - theta) <= angle_tolerance
            # return theta<=angle_tolerance

        def criteria_check(cyl1, cyl2, angle_tolerance, search_angle):
            """
            Decides if cyl2 should be joined to cyl1 and if they are the same tree.
            angle_tolerance is the maximum angle between normal vectors of cylinders to be considered the same branch.
            """
            vector_array = cyl2[:, :3] - np.atleast_2d(cyl1[:3])
            condition1 = within_angle_tolerance(cyl1[3:6], cyl2[:, 3:6], angle_tolerance)
            condition2 = within_angle_tolerance(cyl1[3:6], vector_array, search_angle)
            # condition3 = cyl2[:, self.cyl_dict['radius']] < cyl1[self.cyl_dict['radius']]*1.05
            # cyl2[np.logical_and(condition1, condition2, condition3), self.cyl_dict['tree_id']] = cyl1[self.cyl_dict['tree_id']]
            # cyl2[np.logical_and(condition1, condition2, condition3), self.cyl_dict['parent_branch_id']] = cyl1[self.cyl_dict['branch_id']]
            cyl2[np.logical_and(condition1, condition2), self.cyl_dict["tree_id"]] = cyl1[self.cyl_dict["tree_id"]]
            cyl2[np.logical_and(condition1, condition2), self.cyl_dict["parent_branch_id"]] = cyl1[
                self.cyl_dict["branch_id"]
            ]

            return cyl2

        max_tree_label = 1

        cylinder_array = cylinder_array[
            cylinder_array[:, self.cyl_dict["radius"]] != 0
        ]  # ignore all points with radius of 0.
        unsorted_points = cylinder_array

        sorted_points = np.zeros((0, unsorted_points.shape[1]))
        total_points = len(unsorted_points)
        while unsorted_points.shape[0] > 1:
            if sorted_points.shape[0] % 200 == 0:
                print("\r", np.around(sorted_points.shape[0] / total_points, 3), end="")

            current_point_index = np.argmin(unsorted_points[:, 2])
            current_point = unsorted_points[current_point_index]
            if current_point[self.cyl_dict["tree_id"]] == 0:
                current_point[self.cyl_dict["tree_id"]] = max_tree_label
                max_tree_label += 1

            sorted_points = np.vstack((sorted_points, current_point))
            unsorted_points = np.vstack(
                (unsorted_points[:current_point_index], unsorted_points[current_point_index + 1:])
            )
            kdtree = spatial.cKDTree(unsorted_points[:, :3], leafsize=1000)
            results = kdtree.query_ball_point(np.atleast_2d(current_point)[:, :3], r=distance_tolerance)[0]
            unsorted_points[results] = criteria_check(
                current_point, unsorted_points[results], angle_tolerance, search_angle
            )
        print("1.000\n")
        return sorted_points

    @classmethod
    def make_cyl_visualisation(cls, cyl):
        """Creates a 3D point cloud representation of a circle."""
        p = MeasureTree.create_3d_circles_as_points_flat(cyl[0], cyl[1], cyl[2], cyl[6])
        points = MeasureTree.rodrigues_rot(p - cyl[:3], [0, 0, 1], cyl[3:6])
        points = np.hstack((points + cyl[:3], np.zeros((points.shape[0], 8))))
        points[:, -8:] = cyl[-8:]
        return points

    @classmethod
    def points_along_line(cls, x0, y0, z0, x1, y1, z1, resolution=0.05):
        """Creates a point cloud representation of a line."""
        points_per_line = int(np.linalg.norm(np.array([x1, y1, z1]) - np.array([x0, y0, z0])) / resolution)
        Xs = np.atleast_2d(np.linspace(x0, x1, points_per_line)).T
        Ys = np.atleast_2d(np.linspace(y0, y1, points_per_line)).T
        Zs = np.atleast_2d(np.linspace(z0, z1, points_per_line)).T
        return np.hstack((Xs, Ys, Zs))

    @classmethod
    def create_3d_circles_as_points_flat(cls, x, y, z, r, circle_points=15):
        """Creates a point cloud representation of a horizontal circle at coordinates x, y, z. and of radius r.
        Circle points is the number of points to use to represent each circle."""
        angle_between_points = np.linspace(0, 2 * np.pi, circle_points)
        points = np.zeros((0, 3))
        for i in angle_between_points:
            x2 = r * np.cos(i) + x
            y2 = r * np.sin(i) + y
            point = np.array([[x2, y2, z]])
            points = np.vstack((points, point))
        return points

    @classmethod
    def rodrigues_rot(cls, points, vector1, vector2):
        """RODRIGUES ROTATION
        - Rotate given points based on a starting and ending vector
        - Axis k and angle of rotation theta given by vectors n0,n1
        P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))"""
        if points.ndim == 1:
            points = points[np.newaxis, :]

        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = vector2 / np.linalg.norm(vector2)
        k = np.cross(vector1, vector2)
        if np.sum(k) != 0:
            k = k / np.linalg.norm(k)
        theta = np.arccos(np.dot(vector1, vector2))

        P_rot = np.zeros((len(points), 3))
        for i in range(len(points)):
            P_rot[i] = (
                points[i] * np.cos(theta)
                + np.cross(k, points[i]) * np.sin(theta)
                + k * np.dot(k, points[i]) * (1 - np.cos(theta))
            )
        return P_rot

    @classmethod
    def fit_circle_3D(cls, points, V):
        """
        Fits a circle using Random Sample Consensus (RANSAC) to a set of points in a plane perpendicular to vector V.

        Args:
            points: Set of points to fit a circle to using RANSAC.
            V: Axial vector of the cylinder you're fitting.

        Returns:
            cyl_output: numpy array of the format [[x, y, z, x_norm, y_norm, z_norm, radius, CCI, 0, 0, 0, 0, 0, 0]]
        """

        CCI = 0
        r = 0
        P = points[:, :3]
        P_mean = np.mean(P, axis=0)
        P_centered = P - P_mean
        normal = V / np.linalg.norm(V)
        if normal[2] < 0:  # if normal vector is pointing down, flip it around the other way.
            normal = normal * -1

        # Project points to coords X-Y in 2D plane
        P_xy = MeasureTree.rodrigues_rot(P_centered, normal, [0, 0, 1])

        # Fit circle in new 2D coords with RANSAC
        if P_xy.shape[0] >= 20:

            model_robust, inliers = ransac(
                P_xy[:, :2],
                CircleModel,
                min_samples=int(P_xy.shape[0] * 0.3),
                residual_threshold=0.05,
                max_trials=10000,
            )
            xc, yc = model_robust.params[0:2]
            r = model_robust.params[2]
            CCI = MeasureTree.circumferential_completeness_index([xc, yc], r, P_xy[:, :2])

        if CCI < 0.3:
            r = 0
            xc, yc = np.mean(P_xy[:, :2], axis=0)
            CCI = 0

        # Transform circle center back to 3D coords
        cyl_centre = MeasureTree.rodrigues_rot(np.array([[xc, yc, 0]]), [0, 0, 1], normal) + P_mean
        cyl_output = np.array(
            [
                [
                    cyl_centre[0, 0],
                    cyl_centre[0, 1],
                    cyl_centre[0, 2],
                    normal[0],
                    normal[1],
                    normal[2],
                    r,
                    CCI,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ]
        )
        return cyl_output

    def point_cloud_annotations(self, character_size, xpos, ypos, zpos, offset, text):
        """
        Point based text visualisation. Makes text viewable as a point cloud.

        Args:
            character_size:
            xpos: x coord.
            ypos: y coord.
            zpos: z coord.
            offset: Offset for the x coord. Used to shift the text depending on tree radius.
            text: The text to be displayed.

        Returns:
            nx3 point cloud of the text.
        """

        def convert_character_cells_to_points(character):
            character = np.rot90(character, axes=(1, 0))
            index_i = 0
            index_j = 0
            points = np.zeros((0, 3))
            for i in character:
                for j in i:
                    if j == 1:
                        points = np.vstack((points, np.array([[index_i, index_j, 0]])))
                    index_j += 1
                index_j = 0
                index_i += 1

            roll_mat = np.array(
                [[1, 0, 0], [0, np.cos(-np.pi / 4), -np.sin(-np.pi / 4)], [0, np.sin(-np.pi / 4), np.cos(-np.pi / 4)]]
            )
            points = np.dot(points, roll_mat)
            return points

        def get_character(char):
            if char == ":":
                return self.character_viz[self.characters.index("semiC")]
            elif char == ".":
                return self.character_viz[self.characters.index("dot")]
            elif char == " ":
                return self.character_viz[self.characters.index("space")]
            elif char == "M":
                return self.character_viz[self.characters.index("_M")]
            else:
                return self.character_viz[self.characters.index(char)]

        text_points = np.zeros((11, 0))
        for i in text:
            text_points = np.hstack((text_points, np.array(get_character(str(i)))))
        points = convert_character_cells_to_points(text_points)

        points = points * character_size + [xpos + 0.2 + 0.5 * offset, ypos, zpos]
        return points

    @classmethod
    def fit_cylinder(cls, skeleton_points, point_cloud, num_neighbours):
        """
        Fits a 3D line to the skeleton points cluster provided.
        Uses this line as the major axis/axial vector of the cylinder to be fitted.
        Fits a series of circles perpendicular to this axis to the point cloud of this particular stem segment.

        Args:
            skeleton_points: A single cluster of skeleton points which should represent a segment of a tree/branch.
            point_cloud: The cluster of points belonging to the segment of the branch.
            num_neighbours: The number of skeleton points to use for fitting each circle in the segment. lower numbers
                            have fewer points to fit a circle to, but higher numbers are negatively affected by curved
                            branches. Recommend leaving this as it is.

        Returns:
            cyl_array: a numpy array based representation of the fitted circles/cylinders.
        """
        point_cloud = point_cloud[:, :3]
        skeleton_points = skeleton_points[:, :3]
        cyl_array = np.zeros((0, 14))
        line_centre = np.mean(skeleton_points[:, :3], axis=0)
        _, _, vh = np.linalg.svd(line_centre - skeleton_points)
        line_v_hat = vh[0] / np.linalg.norm(vh[0])

        if skeleton_points.shape[0] <= num_neighbours:
            group = skeleton_points
            line_centre = np.mean([np.min(group[:, :3], axis=0), np.max(group[:, :3], axis=0)], axis=0)
            length = np.linalg.norm(np.max(group, axis=0) - np.min(group, axis=0))
            plane_slice = point_cloud[
                np.linalg.norm(abs(line_v_hat * (point_cloud - line_centre)), axis=1) < (length / 2)
            ]  # calculate distances to plane at centre of line.
            if plane_slice.shape[0] > 0:
                cylinder = MeasureTree.fit_circle_3D(plane_slice, line_v_hat)
                cyl_array = np.vstack((cyl_array, cylinder))
        else:
            while skeleton_points.shape[0] > num_neighbours:
                nn = NearestNeighbors()
                nn.fit(skeleton_points)
                starting_point = np.atleast_2d(skeleton_points[np.argmin(skeleton_points[:, 2])])
                group = skeleton_points[nn.kneighbors(starting_point, n_neighbors=num_neighbours)[1][0]]
                line_centre = np.mean(group[:, :3], axis=0)
                length = np.linalg.norm(np.max(group, axis=0) - np.min(group, axis=0))
                plane_slice = point_cloud[
                    np.linalg.norm(abs(line_v_hat * (point_cloud - line_centre)), axis=1) < (length / 2)
                ]  # calculate distances to plane at centre of line.
                if plane_slice.shape[0] > 0:
                    cylinder = MeasureTree.fit_circle_3D(plane_slice, line_v_hat)
                    cyl_array = np.vstack((cyl_array, cylinder))
                skeleton_points = np.delete(skeleton_points, np.argmin(skeleton_points[:, 2]), axis=0)
        return cyl_array

    @classmethod
    def cylinder_cleaning_multithreaded(cls, args):
        def compute_frustum_volume(diameter_1, diameter_2, height):
            radius_1 = 0.5 * diameter_1
            radius_2 = 0.5 * diameter_2
            volume = (1 / 3) * np.pi * height * (radius_1**2 + radius_1 + radius_2**2 + radius_2)
            return volume

        """
        Cylinder Cleaning
        Works on a single tree worth of cylinders at a time.
        Starts at the lowest (z axis) cylinder.
        Finds neighbouring cylinders within "cleaned_measurement_radius".
        If no neighbours are found, cylinder is deleted.
        If neighbours are found, find the neighbour with the highest circumferential completeness index (CCI). This is
        probably the most trustworthy cylinder in the neighbourhood.

        If there are enough neighbours, use those with CCI >= the 30th percentile of CCIs in the neighbourhood.
        Use the medians of x, y, vx, vy, vz, radius as the cleaned cylinder values.
        Use the mean of the z coords of all neighbours for the cleaned cylinder z coord.
        """
        sorted_cylinders, cleaned_measurement_radius, cyl_dict = args
        cleaned_cyls = np.zeros((0, np.shape(sorted_cylinders)[1]))

        tree = BallTree(sorted_cylinders[:, :3])
        _, ind = tree.query(sorted_cylinders[:, :3], k=10)
        radii = np.array([np.median(sorted_cylinders[row, cyl_dict["radius"]]) for row in ind])
        sorted_cylinders[:, cyl_dict["radius"]] = radii

        while sorted_cylinders.shape[0] > 2:
            start_point_idx = np.argmin(sorted_cylinders[:, 2])
            start_point = sorted_cylinders[start_point_idx]
            sorted_cylinders = np.delete(sorted_cylinders, start_point_idx, axis=0)
            kdtree = spatial.cKDTree(sorted_cylinders[:, :3])
            results = kdtree.query_ball_point(start_point[:3], cleaned_measurement_radius)
            neighbours = sorted_cylinders[results]
            neighbours = np.vstack((neighbours, start_point))
            best_cylinder = start_point

            if neighbours.shape[0] > 0:
                if np.max(neighbours[:, cyl_dict["CCI"]]) > 0:
                    best_cylinder = neighbours[np.argsort(neighbours[:, cyl_dict["CCI"]])][
                        -1
                    ]  # choose cyl with highest CCI.
                # compute 50th percentile of CCI in neighbourhood
                percentile_thresh = np.percentile(neighbours[:, cyl_dict["CCI"]], 50)
                if neighbours[neighbours[:, cyl_dict["CCI"]] >= percentile_thresh, :2].shape[0] > 0:
                    best_cylinder[:3] = np.median(
                        neighbours[neighbours[:, cyl_dict["CCI"]] >= percentile_thresh, :3], axis=0
                    )
                    best_cylinder[3:6] = np.median(
                        neighbours[neighbours[:, cyl_dict["CCI"]] >= percentile_thresh, 3:6], axis=0
                    )
                    best_cylinder[cyl_dict["radius"]] = np.max(
                        neighbours[neighbours[:, cyl_dict["CCI"]] >= percentile_thresh, cyl_dict["radius"]], axis=0
                    )
            cleaned_cyls = np.vstack((cleaned_cyls, best_cylinder))
            sorted_cylinders = np.delete(sorted_cylinders, results, axis=0)
        volume_cyls = deepcopy(cleaned_cyls)
        volume = 0

        while volume_cyls.shape[0] > 2:
            lowest_point_idx = np.argmin(volume_cyls[:, 2])
            lowest_point = volume_cyls[lowest_point_idx]
            volume_cyls = np.delete(volume_cyls, lowest_point_idx, axis=0)
            tree = BallTree(volume_cyls[:, :3], leaf_size=2)
            _, neighbour_idx = tree.query(np.atleast_2d(lowest_point[:3]), 1)
            neighbour_idx = int(neighbour_idx)
            neighbour = volume_cyls[neighbour_idx]
            distance = np.linalg.norm(neighbour[:3] - lowest_point[:3])
            volume += compute_frustum_volume(lowest_point[cyl_dict["radius"]], neighbour[cyl_dict["radius"]], distance)
        cleaned_cyls[:, cyl_dict["tree_volume"]] = volume
        return cleaned_cyls

    @staticmethod
    def inside_conv_hull(point, hull, tolerance=1e-5):
        """Checks if a point is inside a convex hull."""
        return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)

    @classmethod
    def circumferential_completeness_index(cls, fitted_circle_centre, estimated_radius, slice_points):
        """
        Computes the Circumferential Completeness Index (CCI) of a fitted circle.

        Args:
            fitted_circle_centre: x, y coords of the circle centre
            estimated_radius: circle radius
            slice_points: the points the circle was fitted to

        Returns:
            CCI
        """
        sector_angle = 4.5  # degrees
        num_sections = int(np.ceil(360 / sector_angle))
        sectors = np.linspace(-180, 180, num=num_sections, endpoint=False)

        centre_vectors = slice_points[:, :2] - fitted_circle_centre
        norms = np.linalg.norm(centre_vectors, axis=1)

        centre_vectors = centre_vectors / np.atleast_2d(norms).T
        centre_vectors = centre_vectors[
            np.logical_and(norms >= 0.8 * estimated_radius, norms <= 1.2 * estimated_radius)
        ]

        sector_vectors = np.vstack((np.cos(sectors), np.sin(sectors))).T
        CCI = (
            np.sum(
                [
                    np.any(
                        np.degrees(
                            np.arccos(
                                np.clip(np.einsum("ij,ij->i", np.atleast_2d(sector_vector), centre_vectors), -1, 1)
                            )
                        )
                        < sector_angle / 2
                    )
                    for sector_vector in sector_vectors
                ]
            )
            / num_sections
        )

        return CCI

    @classmethod
    def threaded_cyl_fitting(cls, args):
        """Helper function for multithreaded cylinder fitting."""
        skel_cluster, point_cluster, cluster_id, num_neighbours, cyl_dict = args
        cyl_array = np.zeros((0, 14))
        if skel_cluster.shape[0] > num_neighbours:
            cyl_array = cls.fit_cylinder(skel_cluster, point_cluster, num_neighbours=num_neighbours)
            cyl_array[:, cyl_dict["branch_id"]] = cluster_id
        return cyl_array

    @classmethod
    def slice_clustering(cls, new_slice, min_cluster_size):
        """Helper function for clustering stem slices and extracting the skeletons of these stems."""
        cluster_array_internal = np.zeros((0, 6))
        medians = np.zeros((0, 3))

        if new_slice.shape[0] > 1:
            new_slice = cluster_hdbscan(new_slice[:, :3], min_cluster_size)
            for cluster_id in range(0, int(np.max(new_slice[:, -1])) + 1):
                cluster = new_slice[new_slice[:, -1] == cluster_id]
                median = np.median(cluster[:, :3], axis=0)
                medians = np.vstack((medians, median))
                cluster_array_internal = np.vstack(
                    (cluster_array_internal, np.hstack((cluster[:, :3], np.zeros((cluster.shape[0], 3)) + median)))
                )
        return cluster_array_internal, medians

    @classmethod
    def within_angle_tolerances(cls, normal1, normal2, angle_tolerance):
        """Checks if normal1 and normal2 are within "angle_tolerance"
        of each other."""
        norm1 = normal1 / np.atleast_2d(np.linalg.norm(normal1, axis=1)).T
        norm2 = normal2 / np.atleast_2d(np.linalg.norm(normal2, axis=1)).T
        dot = np.clip(np.einsum("ij, ij->i", norm1, norm2), a_min=-1, a_max=1)
        theta = np.degrees(np.arccos(dot))
        return abs((theta > 90) * 180 - theta) <= angle_tolerance

    @classmethod
    def within_search_cone(cls, normal1, vector1_2, search_angle):
        """Checks if the angle between vector1_2 and normal1 is less than search_angle."""
        norm1 = normal1 / np.linalg.norm(normal1)
        if not (vector1_2 == 0).all():
            norm2 = vector1_2 / np.linalg.norm(vector1_2)
            dot = np.clip(np.dot(norm1, norm2), -1, 1)
            theta = math.degrees(np.arccos(dot))
            return abs((theta > 90) * 180 - theta) <= search_angle
        else:
            return False

    def run_measurement_extraction(self):
        taper_array = []
        skeleton_array = np.zeros((0, 3))
        cluster_array = np.zeros((0, 6))
        slice_heights = np.linspace(
            np.min(self.stem_points[:, 2]),
            np.max(self.stem_points[:, 2]),
            int(np.ceil((np.max(self.stem_points[:, 2]) - np.min(self.stem_points[:, 2])) / self.slice_increment)),
        )

        print("Making and clustering slices...")
        i = 0
        max_i = slice_heights.shape[0]
        for slice_height in slice_heights:
            if i % 10 == 0:
                print("\r", i, "/", max_i, end="")
            i += 1
            new_slice = self.stem_points[
                np.logical_and(
                    self.stem_points[:, 2] >= slice_height,
                    self.stem_points[:, 2] < slice_height + self.slice_thickness,
                )
            ]
            if new_slice.shape[0] > 0:
                cluster, skel = MeasureTree.slice_clustering(new_slice, self.parameters["min_cluster_size"])
                skeleton_array = np.vstack((skeleton_array, skel))
                cluster_array = np.vstack((cluster_array, cluster))
        print("\r", max_i, "/", max_i, end="")
        print("\nDone\n")

        print("Clustering skeleton...")
        try:
            skeleton_array = cluster_dbscan(skeleton_array[:, :3], eps=self.slice_increment * 1.5)
        except ValueError:
            raise DataQualityError("Failed to cluster tree skeletons.")
        skeleton_cluster_visualisation = np.zeros((0, 5))
        for k in np.unique(
            skeleton_array[:, -1]
        ):  # Just assigns random colours to the clusters to make it easier to see different neighbouring groups.
            skeleton_cluster_visualisation = np.vstack(
                (
                    skeleton_cluster_visualisation,
                    np.hstack(
                        (
                            skeleton_array[skeleton_array[:, -1] == k],
                            np.zeros((skeleton_array[skeleton_array[:, -1] == k].shape[0], 1))
                            + np.random.randint(0, 10),
                        )
                    ),
                )
            )

        print("Saving skeleton and cluster array...")
        save_file(
            self.output_dir + "skeleton_cluster_visualisation.las",
            skeleton_cluster_visualisation,
            ["X", "Y", "Z", "cluster"],
        )

        print("Making kdtree...")
        # Assign unassigned skeleton points to the nearest group.
        unassigned_bool = skeleton_array[:, -1] == -1
        kdtree = spatial.cKDTree(skeleton_array[unassigned_bool][:, :3], leafsize=100000)
        distances, neighbours = kdtree.query(skeleton_array[unassigned_bool, :3], k=2)
        skeleton_array[unassigned_bool, -1][distances[:, 1] < self.slice_increment * 3] = skeleton_array[
            unassigned_bool, -1
        ][neighbours[:, 1]][distances[:, 1] < self.slice_increment * 3]

        input_data = []
        i = 0
        max_i = int(np.max(skeleton_array[:, -1]) + 1)
        cl_kdtree = spatial.cKDTree(cluster_array[:, 3:], leafsize=100000)
        cluster_ids = range(0, max_i)
        print("Making initial branch/stem section clusters...")

        # organised_clusters = np.zeros((0,5))
        for cluster_id in cluster_ids:
            if i % 100 == 0:
                print("\r", i, "/", max_i, end="")
            i += 1
            skel_cluster = skeleton_array[skeleton_array[:, -1] == cluster_id, :3]
            sc_kdtree = spatial.cKDTree(skel_cluster, leafsize=100000)
            results = np.unique(np.hstack(sc_kdtree.query_ball_tree(cl_kdtree, r=0.0001)))
            cluster_array_clean = cluster_array[results, :3]
            input_data.append(
                [skel_cluster[:, :3], cluster_array_clean[:, :3], cluster_id, self.num_neighbours, self.cyl_dict]
            )

        print("\r", max_i, "/", max_i, end="")
        print("\nDone\n")

        print("Starting multithreaded cylinder fitting... This can take a while.")
        j = 0
        max_j = len(input_data)
        outputlist = []
        with get_context("spawn").Pool(processes=self.num_cpu_cores) as pool:
            for i in pool.imap_unordered(MeasureTree.threaded_cyl_fitting, input_data):
                outputlist.append(i)
                if j % 10 == 0:
                    print("\r", j, "/", max_j, end="")
                j += 1
        full_cyl_array = np.vstack(outputlist)
        print("\r", max_j, "/", max_j, end="")
        print("\nDone\n")

        print("Deleting cyls with CCI less than:", self.parameters["minimum_cci"])
        full_cyl_array = full_cyl_array[full_cyl_array[:, self.cyl_dict["CCI"]] >= self.parameters["minimum_cci"]]

        # cyl_array = [x,y,z,nx,ny,nz,r,CCI,branch_id,tree_id,segment_volume,parent_branch_id]
        print("Saving cylinder array...")
        save_file(self.output_dir + "full_cyl_array.las", full_cyl_array, headers_of_interest=list(self.cyl_dict))
        # full_cyl_array, _ = load_file(self.output_dir + 'full_cyl_array.las',
        #                               headers_of_interest=list(self.cyl_dict))
        if 1:
            print("Making full_cyl visualisation...")
            j = 0
            initial_cyl_vis = []
            max_j = np.shape(full_cyl_array)[0]
            with get_context("spawn").Pool(processes=self.num_cpu_cores) as pool:
                for i in pool.imap_unordered(self.make_cyl_visualisation, full_cyl_array):
                    initial_cyl_vis.append(i)
                    if j % 100 == 0:
                        print("\r", j, "/", max_j, end="")
                    j += 1
            initial_cyl_vis = np.vstack(initial_cyl_vis)
            print("\r", max_j, "/", max_j, end="")
            print("\nDone\n")

            print("\nSaving cylinder visualisation...")
            save_file(self.output_dir + "initial_cyl_vis.las", initial_cyl_vis, headers_of_interest=list(self.cyl_dict))
        print("Sorting Cylinders...")
        full_cyl_array = self.cylinder_sorting(
            full_cyl_array,
            angle_tolerance=self.parameters["sorting_angle_tolerance"],
            search_angle=self.parameters["sorting_search_angle"],
            distance_tolerance=self.parameters["sorting_search_radius"],
        )

        print("Correcting Cylinder assignments...")
        sorted_full_cyl_array = np.zeros((0, full_cyl_array.shape[1]))
        t_id = 1
        max_search_radius = self.parameters["max_search_radius"]
        min_points = 5
        max_search_angle = self.parameters["max_search_angle"]
        max_tree_id = np.unique(full_cyl_array[:, self.cyl_dict["tree_id"]]).shape[0]
        for tree_id in np.unique(full_cyl_array[:, self.cyl_dict["tree_id"]]):
            if int(tree_id) % 10 == 0:
                print("Tree ID", int(tree_id), "/", int(max_tree_id))
            tree = full_cyl_array[full_cyl_array[:, self.cyl_dict["tree_id"]] == int(tree_id)]
            tree_kdtree = spatial.cKDTree(sorted_full_cyl_array[:, :3], leafsize=1000)
            if tree.shape[0] >= min_points:
                lowest_point = tree[np.argmin(tree[:, 2])]
                highest_point = tree[np.argmax(tree[:, 2])]
                lowneighbours = sorted_full_cyl_array[
                    tree_kdtree.query_ball_point(lowest_point[:3], r=max_search_radius)
                ]
                highneighbours = sorted_full_cyl_array[
                    tree_kdtree.query_ball_point(highest_point[:3], r=max_search_radius)
                ]

                lowest_point_z = lowest_point[2] - griddata(
                    (self.DTM[:, 0], self.DTM[:, 1]),
                    self.DTM[:, 2],
                    lowest_point[0:2],
                    method="linear",
                    fill_value=np.median(self.DTM[:, 2]),
                )
                assigned = False
                if lowneighbours.shape[0] > 0:
                    angles = MeasureTree.compute_angle(lowest_point[3:6], lowest_point[:3] - lowneighbours[:, :3])
                    valid_angles = angles[angles <= max_search_angle]

                    if valid_angles.shape[0] > 0:
                        best_parent_point = lowneighbours[np.argmin(angles)]
                        tree = np.vstack(
                            (
                                tree,
                                self.interpolate_cyl(lowest_point, best_parent_point, resolution=self.slice_increment),
                            )
                        )
                        tree[:, self.cyl_dict["tree_id"]] = best_parent_point[self.cyl_dict["tree_id"]]
                        sorted_full_cyl_array = np.vstack((sorted_full_cyl_array, tree))
                        assigned = True
                    else:
                        assigned = False

                elif highneighbours.shape[0] > 0:
                    angles = MeasureTree.compute_angle(highest_point[3:6], highneighbours[:, :3] - highest_point[:3])
                    valid_angles = angles[angles <= max_search_angle]

                    if valid_angles.shape[0] > 0:
                        best_parent_point = highneighbours[np.argmin(angles)]
                        tree = np.vstack(
                            (
                                tree,
                                self.interpolate_cyl(best_parent_point, highest_point, resolution=self.slice_increment),
                            )
                        )
                        tree[:, self.cyl_dict["tree_id"]] = best_parent_point[self.cyl_dict["tree_id"]]
                        sorted_full_cyl_array = np.vstack((sorted_full_cyl_array, tree))
                        assigned = True
                    else:
                        assigned = False

                if assigned is False and lowest_point_z < self.parameters["tree_base_cutoff_height"]:
                    tree[:, self.cyl_dict["tree_id"]] = t_id
                    sorted_full_cyl_array = np.vstack((sorted_full_cyl_array, tree))
                    t_id += 1

        save_file(
            self.output_dir + "sorted_full_cyl_array.las",
            sorted_full_cyl_array,
            headers_of_interest=list(self.cyl_dict),
        )

        print("Cylinder interpolation...")

        tree_list = []
        interpolated_full_cyl_array = np.zeros((0, 14))
        max_tree_id = np.unique(sorted_full_cyl_array[:, self.cyl_dict["tree_id"]]).shape[0]
        for tree_id in np.unique(sorted_full_cyl_array[:, self.cyl_dict["tree_id"]]):
            if int(tree_id) % 10 == 0:
                print("Tree ID", int(tree_id), "/", int(max_tree_id))
            current_tree = sorted_full_cyl_array[sorted_full_cyl_array[:, self.cyl_dict["tree_id"]] == tree_id]
            if current_tree.shape[0] >= self.parameters["min_tree_cyls"]:
                interpolated_full_cyl_array = np.vstack((interpolated_full_cyl_array, current_tree))
                _, individual_branches_indices = np.unique(
                    current_tree[:, self.cyl_dict["branch_id"]], return_index=True
                )
                tree_list.append(nx.Graph())
                for branch in current_tree[individual_branches_indices]:
                    branch_id = branch[self.cyl_dict["branch_id"]]
                    parent_branch_id = branch[self.cyl_dict["parent_branch_id"]]
                    tree_list[-1].add_edge(int(parent_branch_id), int(branch_id))
                    current_branch = current_tree[current_tree[:, self.cyl_dict["branch_id"]] == branch_id]
                    parent_branch = current_tree[current_tree[:, self.cyl_dict["branch_id"]] == parent_branch_id]

                    current_branch_copy = deepcopy(current_branch[np.argsort(current_branch[:, 2])])
                    while current_branch_copy.shape[0] > 1:
                        lowest_point = current_branch_copy[0]
                        current_branch_copy = current_branch_copy[1:]
                        # find nearest point. if nearest point > increment size, interpolate.
                        distances = np.abs(np.linalg.norm(current_branch_copy[:, :3] - lowest_point[:3], axis=1))
                        if distances[distances > 0].shape[0] > 0:
                            if np.min(distances[distances > 0]) > self.slice_increment:
                                interp_to_point = current_branch_copy[distances > 0]
                                if interp_to_point.shape[0] > 0:
                                    interp_to_point = interp_to_point[np.argmin(distances[distances > 0])]

                                # Interpolates a single branch.
                                if interp_to_point.shape[0] > 0:
                                    interpolated_cyls = self.interpolate_cyl(
                                        interp_to_point, lowest_point, resolution=self.slice_increment
                                    )
                                    current_branch = np.vstack((current_branch, interpolated_cyls))
                                    interpolated_full_cyl_array = np.vstack(
                                        (interpolated_full_cyl_array, interpolated_cyls)
                                    )

                    if parent_branch.shape[0] > 0:
                        parent_centre = np.mean(parent_branch[:, :3])
                        closest_point_index = np.argmin(np.linalg.norm(parent_centre - current_branch[:, :3]))
                        closest_point_of_current_branch = current_branch[closest_point_index]
                        kdtree = spatial.cKDTree(parent_branch[:, :3])
                        parent_points_in_range = parent_branch[
                            kdtree.query_ball_point(closest_point_of_current_branch[:3], r=max_search_radius)
                        ]
                        lowest_point_of_current_branch = current_branch[np.argmin(current_branch[:, 2])]
                        if parent_points_in_range.shape[0] > 0:
                            angles = MeasureTree.compute_angle(
                                lowest_point_of_current_branch[3:6],
                                lowest_point_of_current_branch[:3] - parent_points_in_range[:, :3],
                            )
                            angles = angles[angles <= max_search_angle]

                            if angles.shape[0] > 0:
                                best_parent_point = parent_points_in_range[np.argmin(angles)]
                                # Interpolates from lowest point of current branch to smallest angle parent point.
                                interpolated_full_cyl_array = np.vstack(
                                    (
                                        interpolated_full_cyl_array,
                                        self.interpolate_cyl(
                                            lowest_point_of_current_branch,
                                            best_parent_point,
                                            resolution=self.slice_increment,
                                        ),
                                    )
                                )
                current_tree = get_heights_above_DTM(current_tree, self.DTM)
                lowest_10_measured_tree_points = deepcopy(current_tree[np.argsort(current_tree[:, -1])][:10])
                lowest_measured_tree_point = np.median(lowest_10_measured_tree_points, axis=0)
                tree_base_point = deepcopy(current_tree[np.argmin(current_tree[:, self.cyl_dict["height_above_dtm"]])])
                tree_base_point[2] = tree_base_point[2] - tree_base_point[self.cyl_dict["height_above_dtm"]]

                interpolated_to_ground = self.interpolate_cyl(
                    lowest_measured_tree_point, tree_base_point, resolution=self.slice_increment
                )
                interpolated_full_cyl_array = np.vstack((interpolated_full_cyl_array, interpolated_to_ground))

        v1 = interpolated_full_cyl_array[:, 3:6]
        v2 = np.vstack(
            (
                interpolated_full_cyl_array[:, 3],
                interpolated_full_cyl_array[:, 4],
                np.zeros((interpolated_full_cyl_array.shape[0])),
            )
        ).T
        interpolated_full_cyl_array[:, self.cyl_dict["segment_angle_to_horiz"]] = self.compute_angle(v1, v2)
        interpolated_full_cyl_array = get_heights_above_DTM(interpolated_full_cyl_array, self.DTM)

        save_file(
            self.output_dir + "interpolated_full_cyl_array.las",
            interpolated_full_cyl_array,
            headers_of_interest=list(self.cyl_dict),
        )
        # interpolated_full_cyl_array, _ = load_file(self.output_dir + 'interpolated_full_cyl_array.las', headers_of_interest=list(self.cyl_dict))
        print(interpolated_full_cyl_array.shape)
        if 0:
            print("Making interp_cyl visualisation...")
            j = 0
            interpolated_cyl_vis = []
            max_j = np.shape(interpolated_full_cyl_array)[0]
            with get_context("spawn").Pool(processes=self.num_cpu_cores) as pool:
                for i in pool.imap_unordered(self.make_cyl_visualisation, interpolated_full_cyl_array):
                    interpolated_cyl_vis.append(i)
                    if j % 100 == 0:
                        print("\r", j, "/", max_j, end="")
                    j += 1
            interpolated_cyl_vis = np.vstack(interpolated_cyl_vis)
            print("\r", max_j, "/", max_j, end="")
            print("\nDone\n")

            print("\nSaving cylinder visualisation...")
            save_file(
                self.output_dir + "interpolated_cyl_vis.las",
                interpolated_cyl_vis,
                headers_of_interest=list(self.cyl_dict),
            )

        tree_data = np.zeros((0, 16))
        radial_tree_aware_plot_cropping = False
        plot_centre = [[float(self.plot_summary["Plot Centre X"]), float(self.plot_summary["Plot Centre Y"])]]

        stem_points_sorted = np.zeros((0, len(list(self.stem_dict))))
        veg_points_sorted = np.zeros((0, len(list(self.veg_dict))))

        if self.parameters["plot_radius"] > 0 and self.parameters["plot_radius_buffer"] > 0:
            print("Using tree aware plot cropping mode...")
            radial_tree_aware_plot_cropping = True

        print("Cylinder Outlier Removal...")
        input_data = []
        i = 0
        tree_id_list = np.unique(interpolated_full_cyl_array[:, self.cyl_dict["tree_id"]])
        taper_meas_height_max = self.parameters["taper_measurement_height_max"]
        taper_meas_height_min = self.parameters["taper_measurement_height_min"]
        taper_meas_height_increment = self.parameters["taper_measurement_height_increment"]
        self.taper_measurement_heights = np.arange(
            np.floor(taper_meas_height_min * taper_meas_height_increment) / taper_meas_height_increment,
            (np.floor(taper_meas_height_max * taper_meas_height_increment) / taper_meas_height_increment)
            + taper_meas_height_increment,
            taper_meas_height_increment,
        )
        taper_array = np.zeros((0, self.taper_measurement_heights.shape[0] + 5))

        if tree_id_list.shape[0] > 0:
            max_tree_id = int(np.max(tree_id_list))
            for tree_id in tree_id_list:
                if tree_id % 10 == 0:
                    print("\r", tree_id, "/", max_tree_id, end="")
                i += 1
                single_tree = interpolated_full_cyl_array[
                    interpolated_full_cyl_array[:, self.cyl_dict["tree_id"]] == tree_id
                ]
                if single_tree.shape[0] > 0:
                    # single_tree = self.fix_outliers(single_tree)
                    input_data.append([single_tree, self.parameters["cleaned_measurement_radius"], self.cyl_dict])

            print("\r", max_tree_id, "/", max_tree_id, end="")
            print("\nDone\n")

            print("Starting multithreaded cylinder cleaning/smoothing...")
            j = 0
            max_j = len(input_data)

            cleaned_cyls_list = []
            with get_context("spawn").Pool(processes=self.num_cpu_cores) as pool:
                for i in pool.imap_unordered(MeasureTree.cylinder_cleaning_multithreaded, input_data):

                    cleaned_cyls_list.append(i)
                    if j % 11 == 0:
                        print("\r", j, "/", max_j, end="")
                    j += 1
            cleaned_cyls = np.vstack(cleaned_cyls_list)

            del cleaned_cyls_list
            print("\r", max_j, "/", max_j, end="")
            print("\nDone\n")

            save_file(self.output_dir + "cleaned_cyls.las", cleaned_cyls, headers_of_interest=list(self.cyl_dict))
            pd.DataFrame(cleaned_cyls, columns=list(self.cyl_dict)).to_csv(
                self.output_dir + "cleaned_cyls.csv", index=False
            )

            cleaned_cylinders = np.zeros((0, cleaned_cyls.shape[1]))

            print("Sorting vegetation...")
            # Simple nearest neighbours vegetation sorting.
            kdtree = spatial.cKDTree(cleaned_cyls[:, :2], leafsize=1000)
            results = kdtree.query(self.vegetation_points[:, :2], k=1)
            mask = results[0] <= self.parameters["veg_sorting_range"]
            self.vegetation_points = self.vegetation_points[mask]
            results = results[1][mask]
            self.vegetation_points[:, self.veg_dict["tree_id"]] = cleaned_cyls[results, self.cyl_dict["tree_id"]]

            kdtree = spatial.cKDTree(cleaned_cyls[:, :2], leafsize=1000)
            results = kdtree.query(self.stem_points[:, :2], k=1)
            mask = results[0] <= self.parameters["veg_sorting_range"]
            self.stem_points = self.stem_points[mask]
            results = results[1][mask]
            self.stem_points[:, self.stem_dict["tree_id"]] = cleaned_cyls[results, self.cyl_dict["tree_id"]]

            for tree_id in np.unique(cleaned_cyls[:, self.cyl_dict["tree_id"]]):
                tree = cleaned_cyls[cleaned_cyls[:, self.cyl_dict["tree_id"]] == tree_id]
                tree_vegetation = self.vegetation_points[self.vegetation_points[:, self.veg_dict["tree_id"]] == tree_id]
                combined = np.vstack((tree[:, :3], tree_vegetation[:, :3]))
                combined = np.hstack((combined, np.zeros((combined.shape[0], 1))))
                combined = get_heights_above_DTM(combined, self.DTM)

                # Get highest point of tree. Note, there is usually noise, so we use the 98th percentile.
                tree_max_point = combined[
                    abs(
                        combined[:, 2]
                        - np.percentile(combined[:, 2], self.parameters["height_percentile"], interpolation="nearest")
                    ).argmin()
                ]

                tree_base_point = deepcopy(combined[np.argmin(combined[:, -1])])
                z_tree_base = tree_base_point[2] - tree_base_point[-1]

                tree_mean_position = np.mean(combined[:, :2], axis=0)
                tree_height = tree_max_point[-1]
                del combined

                if self.parameters["sort_stems"] or self.parameters["generate_output_point_cloud"]:
                    tree_points = self.stem_points[self.stem_points[:, self.stem_dict["tree_id"]] == tree_id]

                DBH_cyls_slice = tree[
                    np.logical_and(
                        tree[:, self.cyl_dict["height_above_dtm"]] >= 1.0,
                        tree[:, self.cyl_dict["height_above_dtm"]] <= 1.6,
                    )
                ]

                DBH_points_slice = self.stem_points[
                    np.logical_and(
                        self.stem_points[:, self.stem_dict["height_above_dtm"]] >= 1.2,
                        self.stem_points[:, self.stem_dict["height_above_dtm"]] <= 1.4,
                    )
                ]

                DBH = 0
                CCI_at_BH = 0
                DBH_X = 0
                DBH_Y = 0
                DBH_Z = 0

                if DBH_cyls_slice.shape[0] > 0:
                    DBH = np.around(np.mean(DBH_cyls_slice[:, self.cyl_dict["radius"]]) * 2, 3)
                    DBH_X, DBH_Y = np.mean(DBH_cyls_slice[:, :2], axis=0)
                    DBH_Z = z_tree_base + 1.3
                    CCI_at_BH = self.circumferential_completeness_index(
                        [DBH_X, DBH_Y], DBH / 2, DBH_points_slice[:, :2]
                    )

                volume_1 = tree[0, self.cyl_dict["tree_volume"]]
                volume_2 = (
                    np.pi * ((DBH / 2) ** 2) * (((tree_height - 1.3) / 3) + 1.3)
                )  # Volume of a simple vertical cone from DBH to treetop + volume of cylinder from DBH to ground.

                x_tree_base = tree[np.argmin(tree[:, 2]), 0]
                y_tree_base = tree[np.argmin(tree[:, 2]), 1]
                mean_vegetation_density_in_5m_radius = 0
                mean_understory_height_in_5m_radius = 0
                nearby_understory_points = self.ground_veg[self.ground_veg_kdtree.query_ball_point([DBH_X, DBH_Y], r=5)]

                if nearby_understory_points.shape[0] > 0:
                    mean_understory_height_in_5m_radius = np.around(
                        np.nanmean(nearby_understory_points[:, self.veg_dict["height_above_dtm"]]), 2
                    )
                if tree.shape[0] > 0:
                    description = "Tree " + str(int(tree_id))
                    description = description + "\nDBH: " + str(DBH) + " m"
                    description = description + "\nVolume: " + str(np.around(volume_1, 3)) + " m^3"
                    description = description + "\nVolume 2: " + str(np.around(volume_2, 3)) + " m^3"
                    description = description + "\nHeight: " + str(np.around(tree_height, 3)) + " m"
                    description = (
                        description
                        + "\nMean Veg Density (5 m radius): "
                        + str(mean_vegetation_density_in_5m_radius)
                        + " units"
                    )
                    description = (
                        description
                        + "\nMean Understory Height (5 m radius): "
                        + str(mean_understory_height_in_5m_radius)
                        + " m"
                    )

                    print(description)
                    this_trees_data = np.zeros((1, 16), dtype="object")
                    this_trees_data[:, self.tree_data_dict["PlotId"]] = self.plot_summary["PlotId"]
                    this_trees_data[:, self.tree_data_dict["TreeId"]] = int(tree_id)
                    this_trees_data[:, self.tree_data_dict["x_tree_base"]] = x_tree_base
                    this_trees_data[:, self.tree_data_dict["y_tree_base"]] = y_tree_base
                    this_trees_data[:, self.tree_data_dict["z_tree_base"]] = z_tree_base
                    this_trees_data[:, self.tree_data_dict["DBH"]] = DBH
                    this_trees_data[:, self.tree_data_dict["CCI_at_BH"]] = CCI_at_BH
                    this_trees_data[:, self.tree_data_dict["Height"]] = tree_height
                    this_trees_data[:, self.tree_data_dict["Volume_1"]] = volume_1
                    this_trees_data[:, self.tree_data_dict["Volume_2"]] = volume_2
                    this_trees_data[:, self.tree_data_dict["Crown_mean_x"]] = tree_mean_position[0]
                    this_trees_data[:, self.tree_data_dict["Crown_mean_y"]] = tree_mean_position[1]
                    this_trees_data[:, self.tree_data_dict["Crown_top_x"]] = tree_max_point[0]
                    this_trees_data[:, self.tree_data_dict["Crown_top_y"]] = tree_max_point[1]
                    this_trees_data[:, self.tree_data_dict["Crown_top_z"]] = tree_max_point[2]
                    this_trees_data[
                        :, self.tree_data_dict["mean_understory_height_in_5m_radius"]
                    ] = mean_understory_height_in_5m_radius

                    text_size = 0.00256
                    line_height = 0.025
                    if DBH_X != 0 and DBH_Y != 0 and DBH_Z != 0 and x_tree_base != 0 and y_tree_base != 0:
                        tree_id = self.point_cloud_annotations(
                            text_size,
                            DBH_X,
                            DBH_Y + 2 * line_height,
                            DBH_Z + 2 * line_height,
                            DBH * 0.5,
                            "         TREE ID: " + str(int(tree_id)),
                        )
                        line0 = self.point_cloud_annotations(
                            text_size,
                            DBH_X,
                            DBH_Y + line_height,
                            DBH_Z + line_height,
                            DBH * 0.5,
                            "            DIAM: " + str(np.around(DBH, 2)) + "m",
                        )
                        line1 = self.point_cloud_annotations(
                            text_size,
                            DBH_X,
                            DBH_Y,
                            DBH_Z,
                            DBH * 0.5,
                            "       CCI AT BH: " + str(np.around(CCI_at_BH, 2)),
                        )
                        line2 = self.point_cloud_annotations(
                            text_size,
                            DBH_X,
                            DBH_Y - 2 * line_height,
                            DBH_Z - 2 * line_height,
                            DBH * 0.5,
                            "          HEIGHT: " + str(np.around(tree_height, 2)) + "m",
                        )
                        line3 = self.point_cloud_annotations(
                            text_size,
                            DBH_X,
                            DBH_Y - 3 * line_height,
                            DBH_Z - 3 * line_height,
                            DBH * 0.5,
                            "          VOLUME 1: " + str(np.around(volume_1, 2)) + "m3",
                        )
                        line4 = self.point_cloud_annotations(
                            text_size,
                            DBH_X,
                            DBH_Y - 4 * line_height,
                            DBH_Z - 4 * line_height,
                            DBH * 0.5,
                            "          VOLUME 2: " + str(np.around(volume_2, 2)) + "m3",
                        )

                        height_measurement_line = self.points_along_line(
                            x_tree_base,
                            y_tree_base,
                            z_tree_base,
                            x_tree_base,
                            y_tree_base,
                            z_tree_base + tree_height,
                            resolution=0.025,
                        )

                        dbh_circle_points = self.create_3d_circles_as_points_flat(
                            DBH_X, DBH_Y, DBH_Z, DBH / 2, circle_points=100
                        )

                        taper = get_taper(
                            tree,
                            self.taper_measurement_heights,
                            z_tree_base,
                            self.parameters["taper_slice_thickness"],
                            self.plot_summary["PlotId"].item(),
                        )

                        if radial_tree_aware_plot_cropping:
                            if (
                                np.linalg.norm(np.array([x_tree_base, y_tree_base]) - np.array(plot_centre))
                                < self.parameters["plot_radius"]
                            ):
                                tree_data = np.vstack((tree_data, this_trees_data))
                                if self.parameters["sort_stems"] or self.parameters["generate_output_point_cloud"]:
                                    stem_points_sorted = np.vstack(
                                        (stem_points_sorted, tree_points)
                                    )  # TODO make this separate loop as it will be faster.
                                veg_points_sorted = np.vstack((veg_points_sorted, tree_vegetation))
                                cleaned_cylinders = np.vstack((cleaned_cylinders, tree))
                                self.text_point_cloud = np.vstack(
                                    (
                                        self.text_point_cloud,
                                        tree_id,
                                        line0,
                                        line1,
                                        line2,
                                        line3,
                                        line4,
                                        height_measurement_line,
                                        dbh_circle_points,
                                    )
                                )
                                taper_array = np.vstack((taper_array, taper))

                        else:
                            tree_data = np.vstack((tree_data, this_trees_data))
                            taper_array = np.vstack((taper_array, taper))
                            if self.parameters["sort_stems"] or self.parameters["generate_output_point_cloud"]:
                                stem_points_sorted = np.vstack((stem_points_sorted, tree_points))
                            veg_points_sorted = np.vstack((veg_points_sorted, tree_vegetation))
                            cleaned_cylinders = np.vstack((cleaned_cylinders, tree))
                            self.text_point_cloud = np.vstack(
                                (
                                    self.text_point_cloud,
                                    tree_id,
                                    line0,
                                    line1,
                                    line2,
                                    line3,
                                    height_measurement_line,
                                    dbh_circle_points,
                                )
                            )

            save_file(self.output_dir + "text_point_cloud.las", self.text_point_cloud)
            if self.parameters["sort_stems"] or self.parameters["generate_output_point_cloud"]:
                if not self.parameters["minimise_output_size_mode"]:
                    save_file(
                        self.output_dir + "stem_points_sorted.las",
                        stem_points_sorted,
                        headers_of_interest=list(self.stem_dict),
                    )

            if not self.parameters["minimise_output_size_mode"]:
                save_file(
                    self.output_dir + "veg_points_sorted.las",
                    veg_points_sorted,
                    headers_of_interest=list(self.veg_dict),
                )

            if 1:
                print("Making cleaned cylinder visualisation...")
                j = 0
                cleaned_cyl_vis = []
                max_j = np.shape(cleaned_cylinders)[0]
                with get_context("spawn").Pool(processes=self.num_cpu_cores) as pool:
                    for i in pool.imap_unordered(self.make_cyl_visualisation, cleaned_cylinders):
                        cleaned_cyl_vis.append(i)
                        if j % 100 == 0:
                            print("\r", j, "/", max_j, end="")
                        j += 1
                cleaned_cyl_vis = np.vstack(cleaned_cyl_vis)
                print("\r", max_j, "/", max_j, end="")
                print("\nDone\n")

                print("\nSaving cylinder visualisation...")
                save_file(
                    self.output_dir + "cleaned_cyl_vis.las", cleaned_cyl_vis, headers_of_interest=list(self.cyl_dict)
                )

        if radial_tree_aware_plot_cropping and self.parameters["generate_output_point_cloud"]:
            self.terrain_points = self.terrain_points[
                np.linalg.norm(self.terrain_points[:, :2] - plot_centre, axis=1) < self.parameters["plot_radius"]
            ]
            self.cwd_points = self.cwd_points[
                np.linalg.norm(self.cwd_points[:, :2] - plot_centre, axis=1) < self.parameters["plot_radius"]
            ]
            self.ground_veg = self.ground_veg[
                np.linalg.norm(self.ground_veg[:, :2] - plot_centre, axis=1) < self.parameters["plot_radius"]
            ]

            self.DTM = self.DTM[np.linalg.norm(self.DTM[:, :2] - plot_centre, axis=1) < self.parameters["plot_radius"]]
            save_file(self.output_dir + "cropped_DTM.las", self.DTM)
            tree_aware_cropped_point_cloud = np.vstack(
                (self.terrain_points, self.cwd_points, self.ground_veg, stem_points_sorted, veg_points_sorted)
            )  # , stem_points_sorted, veg_points_sorted))

            save_file(
                self.output_dir + "tree_aware_cropped_point_cloud.las",
                tree_aware_cropped_point_cloud,
                headers_of_interest=list(self.stem_dict),
            )

        elif self.parameters["generate_output_point_cloud"]:
            tree_aware_cropped_point_cloud = np.vstack(
                (self.terrain_points, self.cwd_points, self.ground_veg, stem_points_sorted, veg_points_sorted)
            )
            save_file(
                self.output_dir + "tree_aware_cropped_point_cloud.las",
                tree_aware_cropped_point_cloud,
                headers_of_interest=list(self.stem_dict),
            )

        taper_data = pd.DataFrame(
            taper_array,
            columns=["PlotId", "TreeId", "x_base", "y_base", "z_base"]
            + [str(i) for i in self.taper_measurement_heights],
        )
        taper_data.to_csv(self.output_dir + "taper_data.csv", index=None, sep=",")

        tree_data = pd.DataFrame(tree_data, columns=list(self.tree_data_dict))
        tree_data.to_csv(self.output_dir + "tree_data.csv", index=None, sep=",")

        plane = Plane.best_fit(self.DTM)
        avg_gradient = self.compute_angle(plane.normal, [0, 0, 1])
        avg_gradient_x = self.compute_angle(plane.normal[[0, 2]], [0, 1])
        avg_gradient_y = self.compute_angle(plane.normal[[1, 2]], [0, 1])
        self.measure_time_end = time.time()
        self.measure_total_time = self.measure_time_end - self.measure_time_start

        self.plot_summary["Measurement Time (s)"] = self.measure_total_time
        self.plot_summary["Total Run Time (s)"] = (
            self.plot_summary["Preprocessing Time (s)"]
            + self.plot_summary["Semantic Segmentation Time (s)"]
            + self.plot_summary["Post processing time (s)"]
            + self.plot_summary["Measurement Time (s)"]
        )

        self.plot_summary["Num Trees in Plot"] = tree_data.shape[0]
        self.plot_summary["Stems/ha"] = np.around(tree_data.shape[0] / self.plot_area, 1)

        if tree_data.shape[0] > 0:
            self.plot_summary["Mean DBH"] = np.mean(tree_data["DBH"])
            self.plot_summary["Median DBH"] = np.median(tree_data["DBH"])
            self.plot_summary["Min DBH"] = np.min(tree_data["DBH"])
            self.plot_summary["Max DBH"] = np.max(tree_data["DBH"])

            self.plot_summary["Mean Height"] = np.mean(tree_data["Height"])
            self.plot_summary["Median Height"] = np.median(tree_data["Height"])
            self.plot_summary["Min Height"] = np.min(tree_data["Height"])
            self.plot_summary["Max Height"] = np.max(tree_data["Height"])

            self.plot_summary["Mean Volume 1"] = np.mean(tree_data["Volume_1"])
            self.plot_summary["Median Volume 1"] = np.median(tree_data["Volume_1"])
            self.plot_summary["Min Volume 1"] = np.min(tree_data["Volume_1"])
            self.plot_summary["Max Volume 1"] = np.max(tree_data["Volume_1"])
            self.plot_summary["Total Volume 1"] = np.sum(tree_data["Volume_1"])

            self.plot_summary["Mean Volume 2"] = np.mean(tree_data["Volume_2"])
            self.plot_summary["Median Volume 2"] = np.median(tree_data["Volume_2"])
            self.plot_summary["Min Volume 2"] = np.min(tree_data["Volume_2"])
            self.plot_summary["Max Volume 2"] = np.max(tree_data["Volume_2"])
            self.plot_summary["Total Volume 2"] = np.sum(tree_data["Volume_2"])

            self.plot_summary["Canopy Cover Fraction"] = self.canopy_area / self.ground_area

        else:
            self.plot_summary["Mean DBH"] = 0
            self.plot_summary["Median DBH"] = 0
            self.plot_summary["Min DBH"] = 0
            self.plot_summary["Max DBH"] = 0

            self.plot_summary["Mean Height"] = 0
            self.plot_summary["Median Height"] = 0
            self.plot_summary["Min Height"] = 0
            self.plot_summary["Max Height"] = 0

            self.plot_summary["Mean Volume"] = 0
            self.plot_summary["Median Volume"] = 0
            self.plot_summary["Min Volume"] = 0
            self.plot_summary["Max Volume"] = 0
            self.plot_summary["Canopy Cover Fraction"] = 0

        self.plot_summary["Avg Gradient"] = avg_gradient
        self.plot_summary["Avg Gradient X"] = avg_gradient_x
        self.plot_summary["Avg Gradient Y"] = avg_gradient_y

        self.plot_summary["Understory Veg Coverage Fraction"] = float(self.ground_veg_area) / float(self.ground_area)
        self.plot_summary["CWD Coverage Fraction"] = float(self.cwd_area) / float(self.ground_area)

        self.plot_summary.to_csv(self.output_dir + "plot_summary.csv", index=False)
        print("Measuring plot took", self.measure_total_time, "s")
        print("Measuring plot done.")
