import sys
from scripts import run
from scripts import run_with_multiple_plot_centres
import multiple_plot_centres_file_config

if __name__ == '__main__':
    """
    Choose one of the following or modify as needed.
    Directory mode will find all .las files within a directory and sub directories but will ignore any .las files in
    folders with "FSCT_output" in their names.
    File mode will allow you to select multiple .las files within a directory.
    Alternatively, you can just list the point cloud file paths.
    If you have multiple point clouds and wish to enter plot coords for each, have a look at "run_with_multiple_plot_centres.py"
    """
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    mode = "-a"
    type = run

    # If multiple or single processing
    if "-m" in opts:
        type = run_with_multiple_plot_centres
        # args indicates multiple_plot_centres_file_config
        if args is None:
            args = multiple_plot_centres_file_config
    # directory mode
    if "-d" in opts:
        mode = "-d"
    # file mode
    elif "-f" in opts:
        mode = "-f"
    # attended user file mode
    elif "-a" in opts:
        mode = "-a"
    else:
        raise SystemExit(f"Usage: {sys.argv[0]} (-c | -u | -l) <arguments>...")
    type.exec(mode, args)
