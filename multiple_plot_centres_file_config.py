'''
This script is an example of how to provide multiple different plot centres with your input point clouds.
@args
[[*.las, [your_plot_centre_X_coord, your_plot_centre_Y_coord], your_plot_radius], [*.las,[X,Y], radius], [...], [...]]
'''
multiple_plot_centres_file_config = [
  ['your_point_cloud1.las', [0, 0], 100],
  ['your_point_cloud2.las', [300, 200], 50],
]
