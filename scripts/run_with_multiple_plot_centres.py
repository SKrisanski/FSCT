from scripts.run_tools import FSCT
import configparser


def exec(mode, args):
    """
    This script is an example of how to provide multiple different plot centres with your input point clouds.
    @args
    [[*.las, [X,Y], radius], [...], [...]]
    """
    point_clouds_to_process = args
    config_file = configparser.ConfigParser()
    config_file.read("config.ini")
    main_parameters = config_file['FSCT_main_parameters']
    other_parameters = config_file['FSCT_other_parameters']
    for point_cloud_filename, plot_centre, plot_radius in point_clouds_to_process:
        parameters = dict(point_cloud_filename=point_cloud_filename)
        parameters.update(main_parameters)
        parameters.update(other_parameters),
        parameters.update(plot_centre=plot_centre, plot_radius=plot_radius)
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
