from run_tools import FSCT, directory_mode, file_mode
from other_parameters import other_parameters



if __name__ == "__main__":
    """Choose one of the following or modify as needed.
    Directory mode will find all .las files within a directory and sub directories but will ignore any .las files in
    folders with "FSCT_output" in their names.

    File mode will allow you to select multiple .las files within a directory.

    Alternatively, you can just list the point cloud file paths.

    If you have multiple point clouds and wish to enter plot coords for each, have a look at "run_with_multiple_plot_centres.py"
    """
    # point_clouds_to_process = directory_mode()
    # point_clouds_to_process = ['full_path_to_your_point_cloud.las', 'full_path_to_your_second_point_cloud.las', etc.]
    point_clouds_to_process = directory_mode()
    for point_cloud_filename in point_clouds_to_process:
        parameters = dict(
            point_cloud_filename = point_cloud_filename,
            # Adjust if needed
            plot_centre=None,  # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is computed based on the point cloud bounding box.
            # Circular Plot options - Leave at 0 if not using.
            plot_radius=5,  # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
            plot_radius_buffer=5,  # See README. If non-zero, this is used for "Tree Aware Plot Cropping Mode".
            # Set these appropriately for your hardware.
            batch_size=2,  # You will get CUDA errors if this is too high, as you will run out of VRAM. This won't be an issue if running on CPU only. Must be >= 2.
            num_cpu_cores=0,  # Number of CPU cores you want to use. If you run out of RAM, lower this. 0 means ALL cores.
            use_CPU_only=True,  # Set to True if you do not have an Nvidia GPU, or if you don't have enough vRAM.
            # Optional settings - Generally leave as they are.
            slice_thickness=0.10,  # If your point cloud resolution is a bit low (and only if the stem segmentation is still reasonably accurate), try increasing this to 0.2.
            # If your point cloud is really dense, you may get away with 0.1.
            slice_increment=0.05,  # The smaller this is, the better your results will be, however, this increases the run time.
            sort_stems=1,  # If you don't need the sorted stem points, turning this off speeds things up.
            # Veg sorting is required for tree height measurement, but stem sorting isn't necessary for standard use.
            height_percentile=100,  # If the data contains noise above the canopy, you may wish to set this to the 98th percentile of height, otherwise leave it at 100.
            tree_base_cutoff_height=5,  # A tree must have a cylinder measurement below this height above the DTM to be kept. This filters unsorted branches from being called individual trees.
            generate_output_point_cloud=1,  # Turn on if you would like a semantic and instance segmented point cloud. This mode will override the "sort_stems" setting if on.
            # If you activate "tree aware plot cropping mode", this function will use it.
            ground_veg_cutoff_height=1,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
            veg_sorting_range=1.5,  # Vegetation points can be, at most, this far away from a cylinder horizontally to be matched to a particular tree.
            stem_sorting_range=1,  # Stem points can be, at most, this far away from a cylinder in 3D to be matched to a particular tree.
            taper_measurement_height_min=0,  # Lowest height to measure diameter for taper output.
            taper_measurement_height_max=30,  # Highest height to measure diameter for taper output.
            taper_measurement_height_increment=0.2,  # diameter measurement increment.
            taper_slice_thickness=0.4,  # Cylinder measurements within +/- 0.5*taper_slice_thickness are used for taper measurement at a given height. The largest diameter is used.
            delete_working_directory=True,  # Generally leave this on. Deletes the files used for segmentation after segmentation is finished.
            # You may wish to turn it off if you want to re-run/modify the segmentation code so you don't need to run pre-processing every time.
            minimise_output_size_mode=0,  # Will delete a number of non-essential outputs to reduce storage use.
        )

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
