from scripts.run_tools import FSCT, directory_mode, file_mode
from scripts.tools import parse_config
import configparser


def exec(mode, args):
    """
    Choose one of the following or modify as needed.
    Directory mode will find all .las files within a directory and sub directories but will ignore any .las files in
    folders with "FSCT_output" in their names.
    File mode will allow you to select multiple .las files within a directory.
    Alternatively, you can just list the point cloud file paths.
    If you have multiple point clouds and wish to enter plot coords for each, have a look at "run_with_multiple_plot_centres.py"
    """
    if mode == "-d":
        point_clouds_to_process = directory_mode()
    # file mode
    elif mode == "-f":
        point_clouds_to_process = args
    # attended user file mode
    elif mode == "-a":
        point_clouds_to_process = file_mode()

    parser = configparser.ConfigParser()
    parser.read("config.ini")
    main_parameters = parser['FSCT_main_parameters']
    main_parameters = parse_config(main_parameters)
    other_parameters = parser['FSCT_other_parameters']
    other_parameters = parse_config(other_parameters)

    for point_cloud_filename in point_clouds_to_process:
        parameters = dict(point_cloud_filename=point_cloud_filename)
        parameters.update(main_parameters)
        parameters.update(other_parameters)
        FSCT(
            parameters=parameters,
            # Set below to 0 or 1 (or True/False). Each step requires the previous step to have been run already.
            # For standard use, just leave them all set to 1 except "clean_up_files".
            preprocess=1,  # Preparation for semantic segmentation.
            segmentation=1,  # Deep learning based semantic segmentation of the point cloud.
            postprocessing=1,  # Creates the DTM and applies some simple rules to clean up the segmented point cloud.
            measure_plot=1,  # The bulk of the plot measurement happens here.
            make_report=1,  # Generates a plot report, plot map, and some other figures.
            clean_up_files=0,
        )  # Optionally deletes most of the large point cloud outputs to minimise storage requirements.
