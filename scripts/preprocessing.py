import numpy as np
import time
import random
import pandas as pd
import threading
from scripts.tools import load_file, save_file, make_folder_structure, subsample_point_cloud, low_resolution_hack_mode
import os


class Preprocessing:
    def __init__(self, parameters):
        self.preprocessing_time_start = time.time()
        self.parameters = parameters
        self.filename = self.parameters["point_cloud_filename"].replace("\\", "/")
        self.directory = os.path.dirname(os.path.realpath(self.filename)).replace("\\", "/") + "/"
        self.filename = self.filename.split("/")[-1]
        self.box_dimensions = np.array(self.parameters["box_dimensions"])
        self.box_overlap = np.array(self.parameters["box_overlap"])
        self.min_points_per_box = self.parameters["min_points_per_box"]
        self.max_points_per_box = self.parameters["max_points_per_box"]
        self.num_cpu_cores = parameters["num_cpu_cores"]

        self.output_dir, self.working_dir = make_folder_structure(self.directory + self.filename)

        self.point_cloud, headers, self.num_points_orig = load_file(
            filename=self.directory + self.filename,
            plot_centre=self.parameters["plot_centre"],
            plot_radius=self.parameters["plot_radius"],
            plot_radius_buffer=self.parameters["plot_radius_buffer"],
            headers_of_interest=["x", "y", "z", "red", "green", "blue"],
            return_num_points=True,
        )

        self.num_points_trimmed = self.point_cloud.shape[0]

        if self.parameters["plot_centre"] is None:
            mins = np.min(self.point_cloud[:, :2], axis=0)
            maxes = np.max(self.point_cloud[:, :2], axis=0)
            self.parameters["plot_centre"] = (mins + maxes) / 2

        if self.parameters["subsample"]:
            self.point_cloud = subsample_point_cloud(
                self.point_cloud, self.parameters["subsampling_min_spacing"], self.num_cpu_cores
            )

        self.num_points_subsampled = self.point_cloud.shape[0]

        save_file(
            self.output_dir + "working_point_cloud.las",
            self.point_cloud,
            headers_of_interest=["x", "y", "z", "red", "green", "blue"],
        )

        self.point_cloud = self.point_cloud[:, :3]  # Trims off unneeded dimensions if present.

        if self.parameters["low_resolution_point_cloud_hack_mode"]:
            self.point_cloud = low_resolution_hack_mode(
                self.point_cloud,
                self.parameters["low_resolution_point_cloud_hack_mode"],
                self.parameters["subsampling_min_spacing"],
                self.parameters["num_cpu_cores"],
            )

            save_file(self.output_dir + self.filename[:-4] + "_hack_mode_cloud.las", self.point_cloud)

        # Global shift the point cloud to avoid loss of precision during segmentation.
        self.point_cloud[:, :2] = self.point_cloud[:, :2] - self.parameters["plot_centre"]

    @staticmethod
    def threaded_boxes(point_cloud, box_size, min_points_per_box, max_points_per_box, path, id_offset, point_divisions):

        box_centre_mins = point_divisions - 0.5 * box_size
        box_centre_maxes = point_divisions + 0.5 * box_size
        i = 0
        pds = len(point_divisions)
        while i < pds:
            box = point_cloud
            box = box[
                np.logical_and(
                    np.logical_and(
                        np.logical_and(box[:, 0] >= box_centre_mins[i, 0], box[:, 0] < box_centre_maxes[i, 0]),
                        np.logical_and(box[:, 1] >= box_centre_mins[i, 1], box[:, 1] < box_centre_maxes[i, 1]),
                    ),
                    np.logical_and(box[:, 2] >= box_centre_mins[i, 2], box[:, 2] < box_centre_maxes[i, 2]),
                )
            ]

            if box.shape[0] > min_points_per_box:
                if box.shape[0] > max_points_per_box:
                    indices = list(range(0, box.shape[0]))
                    random.shuffle(indices)
                    random.shuffle(indices)
                    box = box[indices[:max_points_per_box], :]
                    box = np.asarray(box, dtype="float64")

                box[:, :3] = box[:, :3]
                np.save(path + str(id_offset + i).zfill(7) + ".npy", box)
            i += 1
        return 1

    def preprocess_point_cloud(self):
        print("Pre-processing point cloud...")
        point_cloud = self.point_cloud  # [self.point_cloud[:,4]!=5]
        Xmax = np.max(point_cloud[:, 0])
        Xmin = np.min(point_cloud[:, 0])
        Ymax = np.max(point_cloud[:, 1])
        Ymin = np.min(point_cloud[:, 1])
        Zmax = np.max(point_cloud[:, 2])
        Zmin = np.min(point_cloud[:, 2])

        X_range = Xmax - Xmin
        Y_range = Ymax - Ymin
        Z_range = Zmax - Zmin

        num_boxes_x = int(np.ceil(X_range / self.box_dimensions[0]))
        num_boxes_y = int(np.ceil(Y_range / self.box_dimensions[1]))
        num_boxes_z = int(np.ceil(Z_range / self.box_dimensions[2]))

        x_vals = np.linspace(
            Xmin, Xmin + (num_boxes_x * self.box_dimensions[0]), int(num_boxes_x / (1 - self.box_overlap[0])) + 1
        )
        y_vals = np.linspace(
            Ymin, Ymin + (num_boxes_y * self.box_dimensions[1]), int(num_boxes_y / (1 - self.box_overlap[1])) + 1
        )
        z_vals = np.linspace(
            Zmin, Zmin + (num_boxes_z * self.box_dimensions[2]), int(num_boxes_z / (1 - self.box_overlap[2])) + 1
        )

        box_centres = np.vstack(np.meshgrid(x_vals, y_vals, z_vals)).reshape(3, -1).T

        point_divisions = []
        for thread in range(self.num_cpu_cores):
            point_divisions.append([])

        points_to_assign = box_centres

        while points_to_assign.shape[0] > 0:
            for i in range(self.num_cpu_cores):
                point_divisions[i].append(points_to_assign[0, :])
                points_to_assign = points_to_assign[1:]
                if points_to_assign.shape[0] == 0:
                    break
        threads = []
        prev_id_offset = 0
        for thread in range(self.num_cpu_cores):
            id_offset = 0
            for t in range(thread):
                id_offset = id_offset + len(point_divisions[t])
            # print('Thread:', thread, prev_id_offset, id_offset)
            prev_id_offset = id_offset
            t = threading.Thread(
                target=Preprocessing.threaded_boxes,
                args=(
                    self.point_cloud,
                    self.box_dimensions,
                    self.min_points_per_box,
                    self.max_points_per_box,
                    self.working_dir,
                    id_offset,
                    point_divisions[thread],
                ),
            )
            threads.append(t)

        for x in threads:
            x.start()

        for x in threads:
            x.join()

        self.preprocessing_time_end = time.time()
        self.preprocessing_time_total = self.preprocessing_time_end - self.preprocessing_time_start
        print("Preprocessing took", self.preprocessing_time_total, "s")
        plot_summary_headers = [
            "PlotId",
            "Point Cloud Filename",
            "Plot Centre X",
            "Plot Centre Y",
            "Plot Radius",
            "Plot Radius Buffer",
            "Plot Area",
            "Num Trees in Plot",
            "Stems/ha",
            "Mean DBH",
            "Median DBH",
            "Min DBH",
            "Max DBH",
            "Mean Height",
            "Median Height",
            "Min Height",
            "Max Height",
            "Mean Volume 1",
            "Median Volume 1",
            "Min Volume 1",
            "Max Volume 1",
            "Total Volume 1",
            "Mean Volume 2",
            "Median Volume 2",
            "Min Volume 2",
            "Max Volume 2",
            "Total Volume 2",
            "Avg Gradient",
            "Avg Gradient X",
            "Avg Gradient Y",
            "Canopy Cover Fraction",
            "Understory Veg Coverage Fraction",
            "CWD Coverage Fraction",
            "Num Points Original PC",
            "Num Points Trimmed PC",
            "Num Points Subsampled PC",
            "Num Terrain Points",
            "Num Vegetation Points",
            "Num CWD Points",
            "Num Stem Points",
            "Preprocessing Time (s)",
            "Semantic Segmentation Time (s)",
            "Post processing time (s)",
            "Measurement Time (s)",
            "Total Run Time (s)",
        ]

        plot_summary = pd.DataFrame(np.zeros((1, len(plot_summary_headers))), columns=plot_summary_headers)

        plot_summary["Preprocessing Time (s)"] = self.preprocessing_time_total
        plot_summary["PlotId"] = self.filename[:-4]
        plot_summary["Point Cloud Filename"] = self.parameters["point_cloud_filename"]
        plot_summary["Plot Centre X"] = self.parameters["plot_centre"][0]
        plot_summary["Plot Centre Y"] = self.parameters["plot_centre"][1]
        plot_summary["Plot Radius"] = self.parameters["plot_radius"]
        plot_summary["Plot Radius Buffer"] = self.parameters["plot_radius_buffer"]
        plot_summary["Num Points Original PC"] = self.num_points_orig
        plot_summary["Num Points Trimmed PC"] = self.num_points_trimmed
        plot_summary["Num Points Subsampled PC"] = self.num_points_subsampled

        plot_summary.to_csv(self.output_dir + "plot_summary.csv", index=False)
        print("Preprocessing done\n")
