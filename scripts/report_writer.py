from mdutils.mdutils import MdUtils
import markdown
from mdutils import Html
import pandas as pd
import numpy as np
from scripts.tools import load_file
from matplotlib import pyplot as plt
import os
from matplotlib.lines import Line2D
import warnings
import shutil


class ReportWriter:
    def __init__(self, parameters):
        self.parameters = parameters
        self.filename = self.parameters["point_cloud_filename"].replace("\\", "/")
        self.output_dir = (
            os.path.dirname(os.path.realpath(self.filename)).replace("\\", "/")
            + "/"
            + self.filename.split("/")[-1][:-4]
            + "_FSCT_output/"
        )
        self.filename = self.filename.split("/")[-1]
        self.plot_summary = pd.read_csv(self.output_dir + "plot_summary.csv", index_col=False)

        self.parameters["plot_centre"] = [
            float(self.plot_summary["Plot Centre X"]),
            float(self.plot_summary["Plot Centre X"]),
        ]

        self.plot_area = float(self.plot_summary["Plot Area"])
        self.stems_per_ha = int(self.plot_summary["Stems/ha"])
        self.parameters["plot_radius"] = float(self.plot_summary["Plot Radius"])
        self.parameters["plot_radius_buffer"] = float(self.plot_summary["Plot Radius Buffer"])

        self.tree_data = pd.read_csv(self.output_dir + "tree_data.csv")
        self.TreeId = np.array(self.tree_data["TreeId"])
        self.x_tree_base = np.array(self.tree_data["x_tree_base"])
        self.y_tree_base = np.array(self.tree_data["y_tree_base"])
        self.DBH = np.array(self.tree_data["DBH"])
        self.height = np.array(self.tree_data["Height"])
        self.Volume_1 = np.array(self.tree_data["Volume_1"])
        self.Volume_2 = np.array(self.tree_data["Volume_2"])

    def make_report(self):
        self.plot_outputs()
        self.create_report()

    def clean_up_files(self):
        files_to_delete = [
            "terrain_points.las",
            "vegetation_points.las",
            "cwd_points.las",
            "stem_points.las",
            "segmented.las",
            "ground_veg.las",
            "Plot_Report.md",
            "working_point_cloud.las",
            "tree_aware_cropped_point_cloud.las",
            "segmented_cleaned.las",
            "veg_points_sorted.las",
            "stem_points_sorted.las",
        ]

        for file in files_to_delete:
            try:
                os.remove(self.output_dir + file)
                print(self.output_dir + file, " deleted.")

            except FileNotFoundError or OSError:
                print(self.output_dir + file, " not found.")

        if self.parameters["delete_working_directory"]:
            shutil.rmtree(self.output_dir + "working_directory/", ignore_errors=True)

    def create_report(self):
        filename = self.output_dir + "Plot_Report"
        mdFile = MdUtils(file_name=filename, title="Forest Structural Complexity Tool - Plot Report")
        mdFile.new_header(level=1, title="")  # style is set 'atx' format by default.
        level = 2

        mdFile.new_header(level=level, title="Point Cloud Filename: " + self.filename)

        mdFile.new_header(
            level=level,
            title="Plot Centre (Local coords): X: "
            + str(np.around(self.parameters["plot_centre"][0], 2))
            + " m, Y: "
            + str(np.around(self.parameters["plot_centre"][1], 2))
            + " m",
        )
        if self.parameters["plot_radius"] != 0:
            mdFile.new_header(
                level=level,
                title="Plot Radius: "
                + str(self.parameters["plot_radius"])
                + " m, "
                + " Plot Radius Buffer: "
                + str(self.parameters["plot_radius_buffer"])
                + " m, Plot Area: "
                + str(np.around(self.plot_area, 3))
                + " ha",
            )
        else:
            mdFile.new_header(level=level, title="Plot Area: " + str(np.around(self.plot_area, 3)) + " ha")

        if self.DBH.shape[0] > 0:
            mdFile.new_header(level=level, title="Stems/ha:  " + str(self.stems_per_ha))
            mdFile.new_header(level=level, title="Mean DBH: " + str(np.around(np.mean(self.DBH), 3)) + " m")
            mdFile.new_header(level=level, title="Median DBH: " + str(np.around(np.median(self.DBH), 3)) + " m")
            mdFile.new_header(level=level, title="Min DBH: " + str(np.around(np.min(self.DBH), 3)) + " m")
            mdFile.new_header(level=level, title="Max DBH: " + str(np.around(np.max(self.DBH), 3)) + " m")

            mdFile.new_header(
                level=level, title="Total Plot Stem Volume 1: " + str(np.around(np.sum(self.Volume_1), 3)) + " m3"
            )
            mdFile.new_header(
                level=level, title="Total Plot Stem Volume 2: " + str(np.around(np.sum(self.Volume_2), 3)) + " m3"
            )

        else:
            mdFile.new_header(level=level, title="Stems/ha: 0")
            mdFile.new_header(level=level, title="No stems found.")

        total_processing_time = float(self.plot_summary["Total Run Time (s)"])

        mdFile.new_header(
            level=level, title="FSCT Processing Time: " + str(np.around(total_processing_time / 60.0, 1)) + " minutes"
        )

        path = "Stem_Map.png"
        mdFile.new_paragraph(Html.image(path=path, size="1000"))

        path = "Diameter at Breast Height Distribution.png"
        mdFile.new_paragraph(Html.image(path=path, size="1000"))

        path = "Tree Height Distribution.png"
        mdFile.new_paragraph(Html.image(path=path, size="1000"))

        path = "Tree Volume 1 Distribution.png"
        mdFile.new_paragraph(Html.image(path=path, size="1000"))

        path = "Tree Volume 2 Distribution.png"
        mdFile.new_paragraph(Html.image(path=path, size="1000"))

        mdFile.create_md_file()

        markdown.markdownFromFile(input=filename + ".md", output=filename + ".html", encoding="utf8")

    def plot_outputs(self):
        self.DTM, _ = load_file(self.output_dir + "DTM.las")
        self.cwd_points, _ = load_file(self.output_dir + "cwd_points.las")
        self.veg_dict = dict(x=0, y=1, z=2, red=3, green=4, blue=5, tree_id=6, height_above_dtm=7)
        self.ground_veg, _ = load_file(self.output_dir + "ground_veg.las", headers_of_interest=list(self.veg_dict))

        self.ground_veg_map = self.ground_veg[:, [0, 1, self.veg_dict["height_above_dtm"]]]
        self.ground_veg_map[self.ground_veg[:, self.veg_dict["height_above_dtm"]] >= 0.5, 2] = 1
        self.ground_veg_map[self.ground_veg[:, self.veg_dict["height_above_dtm"]] < 0.5, 2] = 0.5

        dtmmin = np.min(self.DTM[:, :2], axis=0)
        dtmmax = np.max(self.DTM[:, :2], axis=0)
        plot_max_distance = np.max(dtmmax - dtmmin)
        plot_centre = (dtmmin + dtmmax) / 2
        if self.parameters["plot_centre"] is None:
            self.parameters["plot_centre"] = plot_centre

        fig1 = plt.figure(figsize=(7, 7))
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.set_title("Plot Map - " + self.filename[:-4])
        ax1.set_xlabel("Easting + " + str(np.around(self.parameters["plot_centre"][0], 2)) + " (m)")
        ax1.set_ylabel("Northing + " + str(np.around(self.parameters["plot_centre"][1], 2)) + " (m)")
        ax1.axis("equal")
        zmin = np.floor(np.min(self.DTM[:, 2]))
        zmax = np.ceil(np.max(self.DTM[:, 2]))
        contour_resolution = 1  # metres
        sub_contour_resolution = contour_resolution / 5
        zrange = int(np.ceil((zmax - zmin) / contour_resolution)) + 1
        levels = np.linspace(zmin, zmax, zrange)

        sub_zrange = int(np.ceil((zmax - zmin) / sub_contour_resolution)) + 1
        sub_levels = np.linspace(zmin, zmax, sub_zrange)
        sub_levels = sub_levels[
            sub_levels % contour_resolution != 0
        ]  # remove sub contours where there are full size contours.

        if self.parameters["plot_radius"] != 0:
            ax1.set_facecolor("whitesmoke")
            circle_face = plt.Circle(
                xy=(0, 0), radius=self.parameters["plot_radius"], facecolor="white", edgecolor=None, zorder=1
            )
            ax1.add_patch(circle_face)
            self.ground_veg_map = self.ground_veg_map[
                np.linalg.norm(self.ground_veg_map[:, :2] - plot_centre, axis=1) < self.parameters["plot_radius"]
            ]
            self.cwd_points = self.cwd_points[
                np.linalg.norm(self.cwd_points[:, :2] - plot_centre, axis=1) < self.parameters["plot_radius"]
            ]
            self.DTM = self.DTM[np.linalg.norm(self.DTM[:, :2] - plot_centre, axis=1) < self.parameters["plot_radius"]]

        ax1.scatter(
            self.ground_veg_map[self.ground_veg_map[:, 2] == 0.5, 0] - plot_centre[0],
            self.ground_veg_map[self.ground_veg_map[:, 2] == 0.5, 1] - plot_centre[1],
            marker=".",
            s=4,
            c="greenyellow",
            zorder=3,
        )
        ax1.scatter(
            self.ground_veg_map[self.ground_veg_map[:, 2] == 1, 0] - plot_centre[0],
            self.ground_veg_map[self.ground_veg_map[:, 2] == 1, 1] - plot_centre[1],
            marker=".",
            s=4,
            c="darkseagreen",
            zorder=3,
        )

        circle_outline = plt.Circle(
            xy=(0, 0), radius=self.parameters["plot_radius"], fill=False, edgecolor="k", zorder=3
        )
        ax1.add_patch(circle_outline)

        ax1.scatter(
            self.cwd_points[:, 0] - plot_centre[0],
            self.cwd_points[:, 1] - plot_centre[1],
            marker=".",
            s=1,
            c="yellow",
            zorder=3,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            subcontours = ax1.tricontour(
                self.DTM[:, 0] - plot_centre[0],
                self.DTM[:, 1] - plot_centre[1],
                self.DTM[:, 2],
                levels=sub_levels,
                colors="burlywood",
                linestyles="dashed",
                linewidths=2,
                zorder=3,
            )
            contours = ax1.tricontour(
                self.DTM[:, 0] - plot_centre[0],
                self.DTM[:, 1] - plot_centre[1],
                self.DTM[:, 2],
                levels=levels,
                colors="darkgreen",
                linewidths=2,
                zorder=5,
            )

        plt.clabel(subcontours, inline=True, fmt="%1.1f", fontsize=6, zorder=4)
        plt.clabel(contours, inline=True, fmt="%1.0f", fontsize=10, zorder=6)

        ax1.scatter(
            self.x_tree_base - plot_centre[0],
            self.y_tree_base - plot_centre[1],
            marker=".",
            s=70,
            facecolor="red",
            edgecolor="k",
            zorder=8,
        )
        ax1.scatter([0], [0], marker="x", s=40, c="blue", zorder=9)

        tree_label_offset = np.array([-0.01, 0.01]) * plot_max_distance

        for i in range(0, self.x_tree_base.shape[0]):
            ax1.text(
                (self.x_tree_base[i] - plot_centre[0]) + tree_label_offset[0],
                (self.y_tree_base[i] - plot_centre[1]) + tree_label_offset[1],
                self.TreeId[i],
                fontsize=6,
                zorder=10,
            )

        xmin = np.min(self.DTM[:, 0]) - plot_centre[0]
        xmax = np.max(self.DTM[:, 0]) - plot_centre[0]
        ymin = np.min(self.DTM[:, 1]) - plot_centre[1]
        ymax = np.max(self.DTM[:, 1]) - plot_centre[1]
        padding = 0.1
        ax1.set_xlim([xmin + xmin * padding, xmax + xmax * padding])
        ax1.set_ylim([ymin + ymin * padding, ymax + ymax * padding])

        handles = [
            Line2D(
                range(1),
                range(1),
                label="Understory Veg < 0.5m",
                color="white",
                marker="o",
                markerfacecolor="greenyellow",
                markeredgecolor="lightgrey",
            ),
            Line2D(
                range(1),
                range(1),
                label="Understory Veg >= 0.5m",
                color="white",
                marker="o",
                markerfacecolor="darkseagreen",
                markeredgecolor="lightgrey",
            ),
            Line2D(
                range(1),
                range(1),
                label="Coarse Woody Debris",
                color="white",
                marker="o",
                markerfacecolor="yellow",
                markeredgecolor="lightgrey",
            ),
            Line2D(
                range(1),
                range(1),
                label="Stems",
                color="white",
                marker="o",
                markerfacecolor="red",
                markeredgecolor="k",
            ),
        ]
        ax1.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, facecolor="white")

        fig1.savefig(self.output_dir + "Stem_Map.png", dpi=600, bbox_inches="tight", pad_inches=0.0)
        plt.close()

        fig2 = plt.figure(figsize=(7, 7))
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.set_title("Diameter at Breast Height Distribution", fontsize=10)
        ax2.set_xlabel("DBH (m)")
        ax2.set_ylabel("Count")
        if self.DBH.shape[0] > 0:
            bin_width = 0.1
            bins = np.arange(0, np.ceil(np.max(self.DBH) * 10) / 10 + bin_width, bin_width)

            ax2.hist(
                self.DBH,
                bins=bins,
                range=(0, np.ceil(np.max(self.DBH) * 10) / 10 + bin_width),
                linewidth=0.5,
                edgecolor="black",
                facecolor="green",
            )
        fig2.savefig(
            self.output_dir + "Diameter at Breast Height Distribution.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close()

        fig3 = plt.figure(figsize=(7, 7))
        ax3 = fig3.add_subplot(1, 1, 1)
        ax3.set_title("Tree Height Distribution", fontsize=10)
        ax3.set_xlabel("Height (m)")
        ax3.set_ylabel("Count")
        if self.height.shape[0] > 0:
            bin_width = 1
            bins = np.arange(0, np.ceil(np.max(self.height)) + bin_width, bin_width)

            ax3.hist(
                self.height,
                bins=bins,
                range=(0, np.ceil(np.max(self.height))),
                linewidth=0.5,
                edgecolor="black",
                facecolor="green",
            )
        fig3.savefig(self.output_dir + "Tree Height Distribution.png", dpi=600, bbox_inches="tight", pad_inches=0.0)
        plt.close()

        fig4 = plt.figure(figsize=(7, 7))
        ax4 = fig4.add_subplot(1, 1, 1)
        ax4.set_title("Tree Volume 1 Distribution", fontsize=10)
        ax4.set_xlabel("Volume 1 (m^3)")
        ax4.set_ylabel("Count")
        if self.Volume_1.shape[0] > 0:
            bin_width = 0.1
            bins = np.arange(0, np.ceil(np.max(self.Volume_1) * 10) / 10 + bin_width, bin_width)

            ax4.hist(
                self.Volume_1,
                bins=bins,
                range=(0, np.ceil(np.max(self.Volume_1) * 10) / 10 + bin_width),
                linewidth=0.5,
                edgecolor="black",
                facecolor="green",
            )
        fig4.savefig(self.output_dir + "Tree Volume 1 Distribution.png", dpi=600, bbox_inches="tight", pad_inches=0.0)
        plt.close()

        fig4 = plt.figure(figsize=(7, 7))
        ax4 = fig4.add_subplot(1, 1, 1)
        ax4.set_title("Tree Volume 2 Distribution", fontsize=10)
        ax4.set_xlabel("Volume 2 (m^3)")
        ax4.set_ylabel("Count")
        if self.Volume_2.shape[0] > 0:
            bin_width = 0.1
            bins = np.arange(0, np.ceil(np.max(self.Volume_2) * 10) / 10 + bin_width, bin_width)

            ax4.hist(
                self.Volume_2,
                bins=bins,
                range=(0, np.ceil(np.max(self.Volume_2) * 10) / 10 + bin_width),
                linewidth=0.5,
                edgecolor="black",
                facecolor="green",
            )
        fig4.savefig(self.output_dir + "Tree Volume 2 Distribution.png", dpi=600, bbox_inches="tight", pad_inches=0.0)
        plt.close()
