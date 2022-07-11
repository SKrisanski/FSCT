import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import random
import glob
from sklearn.neighbors import NearestNeighbors


class TrainingDataset(Dataset):
    def __init__(self, root_dir, device, min_sample_points, max_sample_points):
        super().__init__()
        self.filenames = glob.glob(root_dir + "*.npy")
        self.max_sample_points = max_sample_points
        self.label_index = 3
        self.device = device
        self.min_sample_points = min_sample_points

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        point_cloud = np.load(self.filenames[index])
        if point_cloud.shape[0] > self.max_sample_points:
            shuffle_index = list(range(point_cloud.shape[0]))
            random.shuffle(shuffle_index)
            point_cloud = point_cloud[shuffle_index[: self.max_sample_points]]

        x = point_cloud[:, :3]
        y = point_cloud[:, self.label_index] - 1
        x, y = augmentations(x, y, self.min_sample_points)
        if np.all(y != 0):
            y[y == 2] = 3  # if no ground is present, CWD is relabelled as stem.
        x = torch.from_numpy(x.copy()).type(torch.float).to(self.device)
        y = torch.from_numpy(y.copy()).type(torch.long).to(self.device)

        # Place sample at origin
        global_shift = torch.mean(x[:, :3], axis=0)
        x = x - global_shift

        data = Data(pos=x, x=None, y=y)
        return data


class ValidationDataset(Dataset):
    def __init__(self, root_dir, device):
        super().__init__()
        self.filenames = glob.glob(root_dir + "*.npy")
        self.label_index = 3
        self.device = device

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        with torch.no_grad():
            point_cloud = np.load(self.filenames[index])
            x = point_cloud[:, :3]
            y = point_cloud[:, self.label_index] - 1
            x = torch.from_numpy(x.copy()).type(torch.float).to(self.device)
            y = torch.from_numpy(y.copy()).type(torch.long).to(self.device)

            # Place sample at origin
            global_shift = torch.mean(x[:, :3], axis=0)
            x = x - global_shift

            data = Data(pos=x, x=None, y=y)
            return data


def augmentations(x, y, min_sample_points):
    def rotate_3d(points, rotations):
        rotations[0] = np.radians(rotations[0])
        rotations[1] = np.radians(rotations[1])
        rotations[2] = np.radians(rotations[2])

        roll_mat = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rotations[0]), -np.sin(rotations[0])],
                [0, np.sin(rotations[0]), np.cos(rotations[0])],
            ]
        )

        pitch_mat = np.array(
            [
                [np.cos(rotations[1]), 0, np.sin(rotations[1])],
                [0, 1, 0],
                [-np.sin(rotations[1]), 0, np.cos(rotations[1])],
            ]
        )

        yaw_mat = np.array(
            [
                [np.cos(rotations[2]), -np.sin(rotations[2]), 0],
                [np.sin(rotations[2]), np.cos(rotations[2]), 0],
                [0, 0, 1],
            ]
        )

        points[:, :3] = np.matmul(np.matmul(np.matmul(points[:, :3], roll_mat), pitch_mat), yaw_mat)
        return points

    def random_scale_change(points, min_multiplier, max_multiplier):
        points = points * np.random.uniform(min_multiplier, max_multiplier)
        return points

    def random_point_removal(x, y, min_sample_points):
        indices = np.arange(np.shape(x)[0])
        np.random.shuffle(indices)
        num_points_to_keep = min_sample_points + int(np.random.uniform(0, 0.95) * (np.shape(x)[0] - min_sample_points))
        indices = indices[:num_points_to_keep]
        return x[indices], y[indices]

    def random_noise_addition(points):
        # 50% chance per sample of adding noise.
        random_noise_std_dev = np.random.uniform(0.01, 0.025)
        if np.random.uniform(0, 1) >= 0.5:
            points = points + np.random.normal(0, random_noise_std_dev, size=(np.shape(points)[0], 3))
        return points

    if np.all(y != 0) and np.all(
        y != 2
    ):  # if no terrain or CWD are present, it's ok to rotate extremely. Terrain shouldn't be above stems or CWD.
        rotations = [np.random.uniform(-90, 90), np.random.uniform(-90, 90), np.random.uniform(-180, 180)]
    else:
        rotations = [np.random.uniform(-25, 25), np.random.uniform(-25, 25), np.random.uniform(-180, 180)]
    x = rotate_3d(x, rotations)
    x = random_scale_change(x, 0.8, 1.2)
    if np.random.uniform(0, 1) >= 0.5 and x.shape[0] > min_sample_points:
        x, y = subsample_point_cloud(x, y, np.random.uniform(0.01, 0.025), min_sample_points)

    if np.random.uniform(0, 1) >= 0.8 and x.shape[0] > min_sample_points:
        x, y = random_point_removal(x, y, min_sample_points)

    x = random_noise_addition(x)
    return x, y


def subsample_point_cloud(x, y, min_spacing, min_sample_points):
    x = np.hstack((x, np.atleast_2d(y).T))
    neighbours = NearestNeighbors(n_neighbors=2, algorithm="kd_tree", metric="euclidean").fit(x[:, :3])
    distances, indices = neighbours.kneighbors(x[:, :3])
    x_keep = x[distances[:, 1] >= min_spacing]
    i1 = [distances[:, 1] < min_spacing][0]
    i2 = [x[indices[:, 0], 2] < x[indices[:, 1], 2]][0]
    x_check = x[np.logical_and(i1, i2)]

    while np.shape(x_check)[0] > 1:
        neighbours = NearestNeighbors(n_neighbors=2, algorithm="kd_tree", metric="euclidean").fit(x_check[:, :3])
        distances, indices = neighbours.kneighbors(x_check[:, :3])
        x_keep = np.vstack((x_keep, x_check[distances[:, 1] >= min_spacing, :]))
        i1 = [distances[:, 1] < min_spacing][0]
        i2 = [x_check[indices[:, 0], 2] < x_check[indices[:, 1], 2]][0]
        x_check = x_check[np.logical_and(i1, i2)]
    if x_keep.shape[0] >= min_sample_points:
        return x_keep[:, :3], x_keep[:, 3]
    else:
        return x[:, :3], x[:, 3]
