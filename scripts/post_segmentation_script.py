import numpy as np
import pandas as pd
from scipy import spatial
import time
import warnings
import os
from scripts.tools import load_file, save_file, get_heights_above_DTM
from scripts.fsct_exceptions import DataQualityError

warnings.filterwarnings("ignore", category=RuntimeWarning)


class PostProcessing:
    def __init__(self, parameters):
        self.post_processing_time_start = time.time()
        self.parameters = parameters
        self.filename = self.parameters["point_cloud_filename"].replace("\\", "/")
        self.output_dir = (
            os.path.dirname(os.path.realpath(self.filename)).replace("\\", "/")
            + "/"
            + self.filename.split("/")[-1][:-4]
            + "_FSCT_output/"
        )
        self.filename = self.filename.split("/")[-1]

        self.noise_class_label = parameters["noise_class"]
        self.terrain_class_label = parameters["terrain_class"]
        self.vegetation_class_label = parameters["vegetation_class"]
        self.cwd_class_label = parameters["cwd_class"]
        self.stem_class_label = parameters["stem_class"]
        print("Loading segmented point cloud...")
        self.point_cloud, self.headers_of_interest = load_file(
            self.output_dir + "segmented.las", headers_of_interest=["x", "y", "z", "red", "green", "blue", "label"]
        )
        self.point_cloud = np.hstack(
            (self.point_cloud, np.zeros((self.point_cloud.shape[0], 1)))
        )  # Add height above DTM column
        self.headers_of_interest.append("height_above_DTM")  # Add height_above_DTM to the headers.
        self.label_index = self.headers_of_interest.index("label")
        self.point_cloud[:, self.label_index] = (
            self.point_cloud[:, self.label_index] + 1
        )  # index offset since noise_class was removed from inference.
        self.plot_summary = pd.read_csv(self.output_dir + "plot_summary.csv", index_col=None)

    def make_DTM(self, crop_dtm=False):
        print("Making DTM...")
        full_point_cloud_kdtree = spatial.cKDTree(self.point_cloud[:, :2])
        terrain_kdtree = spatial.cKDTree(self.terrain_points[:, :2])
        xmin = np.floor(np.min(self.terrain_points[:, 0])) - 3
        ymin = np.floor(np.min(self.terrain_points[:, 1])) - 3
        xmax = np.ceil(np.max(self.terrain_points[:, 0])) + 3
        ymax = np.ceil(np.max(self.terrain_points[:, 1])) + 3
        x_points = np.linspace(xmin, xmax, int(np.ceil((xmax - xmin) / self.parameters["grid_resolution"])) + 1)
        y_points = np.linspace(ymin, ymax, int(np.ceil((ymax - ymin) / self.parameters["grid_resolution"])) + 1)
        grid_points = np.zeros((0, 3))

        for x in x_points:
            for y in y_points:
                radius = self.parameters["grid_resolution"] * 3
                indices = terrain_kdtree.query_ball_point([x, y], r=radius)
                while len(indices) <= 100 and radius <= self.parameters["grid_resolution"] * 5:
                    radius += self.parameters["grid_resolution"]
                    indices = terrain_kdtree.query_ball_point([x, y], r=radius)
                if len(indices) >= 100:
                    z = np.percentile(self.terrain_points[indices, 2], 20)
                    grid_points = np.vstack((grid_points, np.array([[x, y, z]])))

                else:
                    indices = full_point_cloud_kdtree.query_ball_point([x, y], r=radius)
                    while len(indices) <= 100:
                        radius += self.parameters["grid_resolution"]
                        indices = full_point_cloud_kdtree.query_ball_point([x, y], r=radius)

                    z = np.percentile(self.point_cloud[indices, 2], 2.5)
                    grid_points = np.vstack((grid_points, np.array([[x, y, z]])))

        if self.parameters["plot_radius"] > 0:
            plot_centre = [[float(self.plot_summary["Plot Centre X"]), float(self.plot_summary["Plot Centre Y"])]]
            crop_radius = self.parameters["plot_radius"] + self.parameters["plot_radius_buffer"]
            grid_points = grid_points[np.linalg.norm(grid_points[:, :2] - plot_centre, axis=1) <= crop_radius]

        elif crop_dtm:
            distances, _ = full_point_cloud_kdtree.query(grid_points[:, :2], k=[1])
            distances = np.squeeze(distances)
            grid_points = grid_points[distances <= self.parameters["grid_resolution"]]
        print("    DTM Done")
        return grid_points

    def process_point_cloud(self):
        self.terrain_points = self.point_cloud[
            self.point_cloud[:, self.label_index] == self.terrain_class_label
        ]  # -2 is now the class label as we added the height above DTM column.

        try:
            self.DTM = self.make_DTM(crop_dtm=True)
        except ValueError:
            raise DataQualityError("Failed to make DTM. \nThere probably aren't any terrain_points.")

        save_file(self.output_dir + "DTM.las", self.DTM)

        self.convexhull = spatial.ConvexHull(self.DTM[:, :2])
        self.plot_area = self.convexhull.volume / 10000  # volume is area in 2d.
        print("Plot area is approximately", self.plot_area, "ha")

        above_and_below_DTM_trim_dist = 0.2

        self.point_cloud = get_heights_above_DTM(
            self.point_cloud, self.DTM
        )  # Add a height above DTM column to the point clouds.
        self.terrain_points = self.point_cloud[self.point_cloud[:, self.label_index] == self.terrain_class_label]
        self.terrain_points_rejected = np.vstack(
            (
                self.terrain_points[self.terrain_points[:, -1] <= -above_and_below_DTM_trim_dist],
                self.terrain_points[self.terrain_points[:, -1] > above_and_below_DTM_trim_dist],
            )
        )
        self.terrain_points = self.terrain_points[
            np.logical_and(
                self.terrain_points[:, -1] > -above_and_below_DTM_trim_dist,
                self.terrain_points[:, -1] < above_and_below_DTM_trim_dist,
            )
        ]

        save_file(
            self.output_dir + "terrain_points.las",
            self.terrain_points,
            headers_of_interest=self.headers_of_interest,
            silent=False,
        )
        self.stem_points = self.point_cloud[self.point_cloud[:, self.label_index] == self.stem_class_label]
        self.terrain_points = np.vstack(
            (
                self.terrain_points,
                self.stem_points[
                    np.logical_and(
                        self.stem_points[:, -1] >= -above_and_below_DTM_trim_dist,
                        self.stem_points[:, -1] <= above_and_below_DTM_trim_dist,
                    )
                ],
            )
        )
        self.stem_points_rejected = self.stem_points[self.stem_points[:, -1] <= above_and_below_DTM_trim_dist]
        self.stem_points = self.stem_points[self.stem_points[:, -1] > above_and_below_DTM_trim_dist]
        save_file(
            self.output_dir + "stem_points.las",
            self.stem_points,
            headers_of_interest=self.headers_of_interest,
            silent=False,
        )

        self.vegetation_points = self.point_cloud[self.point_cloud[:, self.label_index] == self.vegetation_class_label]
        self.terrain_points = np.vstack(
            (
                self.terrain_points,
                self.vegetation_points[
                    np.logical_and(
                        self.vegetation_points[:, -1] >= -above_and_below_DTM_trim_dist,
                        self.vegetation_points[:, -1] <= above_and_below_DTM_trim_dist,
                    )
                ],
            )
        )
        self.vegetation_points_rejected = self.vegetation_points[
            self.vegetation_points[:, -1] <= above_and_below_DTM_trim_dist
        ]
        self.vegetation_points = self.vegetation_points[self.vegetation_points[:, -1] > above_and_below_DTM_trim_dist]
        save_file(
            self.output_dir + "vegetation_points.las",
            self.vegetation_points,
            headers_of_interest=self.headers_of_interest,
            silent=False,
        )

        self.cwd_points = self.point_cloud[
            self.point_cloud[:, self.label_index] == self.cwd_class_label
        ]  # -2 is now the class label as we added the height above DTM column.
        self.terrain_points = np.vstack(
            (
                self.terrain_points,
                self.cwd_points[
                    np.logical_and(
                        self.cwd_points[:, -1] >= -above_and_below_DTM_trim_dist,
                        self.cwd_points[:, -1] <= above_and_below_DTM_trim_dist,
                    )
                ],
            )
        )

        self.cwd_points_rejected = np.vstack(
            (
                self.cwd_points[self.cwd_points[:, -1] <= above_and_below_DTM_trim_dist],
                self.cwd_points[self.cwd_points[:, -1] >= 10],
            )
        )
        self.cwd_points = self.cwd_points[
            np.logical_and(self.cwd_points[:, -1] > above_and_below_DTM_trim_dist, self.cwd_points[:, -1] < 3)
        ]
        save_file(
            self.output_dir + "cwd_points.las",
            self.cwd_points,
            headers_of_interest=self.headers_of_interest,
            silent=False,
        )

        self.terrain_points[:, self.label_index] = self.terrain_class_label
        self.cleaned_pc = np.vstack((self.terrain_points, self.vegetation_points, self.cwd_points, self.stem_points))
        save_file(
            self.output_dir + "segmented_cleaned.las", self.cleaned_pc, headers_of_interest=self.headers_of_interest
        )

        self.post_processing_time_end = time.time()
        self.post_processing_time = self.post_processing_time_end - self.post_processing_time_start
        print("Post-processing took", self.post_processing_time, "seconds")
        self.plot_summary["Post processing time (s)"] = self.post_processing_time
        self.plot_summary["Num Terrain Points"] = self.terrain_points.shape[0]
        self.plot_summary["Num Vegetation Points"] = self.vegetation_points.shape[0]
        self.plot_summary["Num CWD Points"] = self.cwd_points.shape[0]
        self.plot_summary["Num Stem Points"] = self.stem_points.shape[0]
        self.plot_summary["Plot Area"] = self.plot_area
        self.plot_summary["Post processing time (s)"] = self.post_processing_time
        self.plot_summary.to_csv(self.output_dir + "plot_summary.csv", index=False)
        print("Post processing done.")
