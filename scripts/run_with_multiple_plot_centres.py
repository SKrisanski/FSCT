from run_tools import FSCT, directory_mode, file_mode
from other_parameters import other_parameters


if __name__ == "__main__":
    """
    This script is an example of how to provide multiple different plot centres with your input point clouds.
    """
    point_clouds_to_process =  []

    for point_cloud_filename, plot_centre, plot_radius in point_clouds_to_process:
        parameters = dict(
            point_cloud_filename=point_cloud_filename,
            plot_centre=plot_centre,
            plot_radius=plot_radius,
            plot_radius_buffer=0,
            batch_size=18,
            num_cpu_cores=0,
            use_CPU_only=True,
            # Optional settings - Generally leave as they are.
            slice_thickness=0.15,
            # If your point cloud resolution is a bit low (and only if the stem segmentation is still reasonably accurate), try increasing this to 0.2.
            # If your point cloud is really dense, you may get away with 0.1.
            slice_increment=0.05,
            # The smaller this is, the better your results will be, however, this increases the run time.
            sort_stems=1,  # If you don't need the sorted stem points, turning this off speeds things up.
            # Veg sorting is required for tree height measurement, but stem sorting isn't necessary for standard use.
            height_percentile=100,
            # If the data contains noise above the canopy, you may wish to set this to the 98th percentile of height, otherwise leave it at 100.
            tree_base_cutoff_height=5,
            # A tree must have a cylinder measurement below this height above the DTM to be kept. This filters unsorted branches from being called individual trees.
            generate_output_point_cloud=1,
            # Turn on if you would like a semantic and instance segmented point cloud. This mode will override the "sort_stems" setting if on.
            # If you activate "tree aware plot cropping mode", this function will use it.
            ground_veg_cutoff_height=2,
            # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
            veg_sorting_range=1.5,
            # Vegetation points can be, at most, this far away from a cylinder horizontally to be matched to a particular tree.
            stem_sorting_range=1,
            # Stem points can be, at most, this far away from a cylinder in 3D to be matched to a particular tree.
            taper_measurement_height_min=0,  # Lowest height to measure diameter for taper output.
            taper_measurement_height_max=30,  # Highest height to measure diameter for taper output.
            taper_measurement_height_increment=0.2,  # diameter measurement increment.
            taper_slice_thickness=0.4,
            # Cylinder measurements within +/- 0.5*taper_slice_thickness are used for taper measurement at a given height. The largest diameter is used.
            delete_working_directory=True,
            # Generally leave this on. Deletes the files used for segmentation after segmentation is finished.
            # You may wish to turn it off if you want to re-run/modify the segmentation code so you don't need to run pre-processing every time.
            minimise_output_size_mode=0,
            # Will delete a number of non-essential outputs to reduce storage use.
        )

        parameters.update(other_parameters)
        FSCT(
            parameters=parameters,
            preprocess=1,
            segmentation=1,
            postprocessing=1,
            measure_plot=1,
            make_report=1,
            clean_up_files=0,
        )
