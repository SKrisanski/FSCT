import pandas as pd

from scripts.run_tools import FSCT, directory_mode, file_mode


def combine_multiple_output_CSVs(point_clouds_to_process, csv_file_to_combine):
    combined_plot_summary_dataframe = pd.read_csv(
        point_clouds_to_process[0][:-4] + "_FSCT_output/" + csv_file_to_combine, index_col=[0]
    )
    if len(point_clouds_to_process) > 1:
        for point_cloud_filename in point_clouds_to_process[1:]:
            report_filename = point_cloud_filename[:-4] + "_FSCT_output/" + csv_file_to_combine
            # print(report_filename)
            combined_plot_summary_dataframe = pd.concat(
                [combined_plot_summary_dataframe, pd.read_csv(report_filename, index_col=[0])]
            )
    return combined_plot_summary_dataframe


def get_lowest_common_directory(point_clouds_to_process):
    path_list = []
    for filepath in point_clouds_to_process:
        path_list.append(filepath.split("/"))

    path_set = set(path_list[0])
    while len(path_list) > 1:
        path_list = path_list[1:]
        path_set = path_set.intersection(path_list[0])
    output_directory = []
    for path in path_list:
        for directory in path:
            if directory in path_set:
                output_directory.append(directory)

    output_directory = "/".join(output_directory)
    return output_directory


if __name__ == "__main__":
    """Choose one of the following or modify as needed.
    Directory mode will find all .las files within a directory and sub directories but will ignore any .las files in
    folders with "FSCT_output" in their names.

    File mode will allow you to select multiple .las files within a directory.

    Alternatively, you can just list the point cloud file paths.

    This will find the outputs of the FSCT processing for all of these point clouds and combine the "plot_reports".
    """
    # point_clouds_to_process = directory_mode()
    point_clouds_to_process = file_mode()

    output_directory = get_lowest_common_directory(point_clouds_to_process)
    combined_plot_summary_dataframe = combine_multiple_output_CSVs(point_clouds_to_process, "plot_summary.csv")
    combined_plot_summary_dataframe.to_csv(output_directory + "/combined_plot_summaries.csv", sep=",")

    combined_tree_data = combine_multiple_output_CSVs(point_clouds_to_process, "tree_data.csv")
    combined_tree_data.to_csv(output_directory + "/combined_tree_data.csv", sep=",")

    combined_taper_data = combine_multiple_output_CSVs(point_clouds_to_process, "taper_data.csv")
    combined_taper_data.to_csv(output_directory + "/combined_taper_data.csv", sep=",")
