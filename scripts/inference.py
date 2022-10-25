from abc import ABC
import torch
import torch_geometric
from torch_geometric.data import Dataset, DataLoader, Data
import numpy as np
import glob
import pandas as pd
from preprocessing import Preprocessing
from model import Net
from sklearn.neighbors import NearestNeighbors
from scipy import spatial
import os
import time
from tools import get_fsct_path
from tools import load_file, save_file
import shutil
import sys

sys.setrecursionlimit(10**8)  # Can be necessary for dealing with large point clouds.


class TestingDataset(Dataset, ABC):
    def __init__(self, root_dir, points_per_box, device):
        super().__init__()
        self.filenames = glob.glob(root_dir + "*.npy")
        self.device = device
        self.points_per_box = points_per_box

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        point_cloud = np.load(self.filenames[index])
        pos = point_cloud[:, :3]
        pos = torch.from_numpy(pos.copy()).type(torch.float).to(self.device).requires_grad_(False)

        # Place sample at origin
        local_shift = torch.round(torch.mean(pos[:, :3], axis=0)).requires_grad_(False)
        pos = pos - local_shift
        data = Data(pos=pos, x=None, local_shift=local_shift)
        return data


def choose_most_confident_label(point_cloud, original_point_cloud):
    """
    Args:
        original_point_cloud: The original point cloud to be labeled.
        point_cloud: The segmented point cloud (often slightly downsampled from the process).

    Returns:
        The original point cloud with segmentation labels added.
    """

    print("Choosing most confident labels...")
    neighbours = NearestNeighbors(n_neighbors=16, algorithm="kd_tree", metric="euclidean", radius=0.05).fit(
        point_cloud[:, :3]
    )
    _, indices = neighbours.kneighbors(original_point_cloud[:, :3])

    labels = np.zeros((original_point_cloud.shape[0], 5))
    labels[:, :4] = np.median(point_cloud[indices][:, :, -4:], axis=1)
    labels[:, 4] = np.argmax(labels[:, :4], axis=1)

    original_point_cloud = np.hstack((original_point_cloud, labels[:, 4:]))
    return original_point_cloud


class SemanticSegmentation:
    def __init__(self, parameters):
        self.sem_seg_start_time = time.time()
        self.parameters = parameters

        if not self.parameters["use_CPU_only"]:
            print("Is CUDA available?", torch.cuda.is_available())
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        print("Performing inference on device:", self.device)
        if not torch.cuda.is_available():
            print("Please be aware that inference will be much slower on CPU. An Nvidia GPU is highly recommended.")
        self.filename = self.parameters["point_cloud_filename"].replace("\\", "/")
        self.directory = os.path.dirname(os.path.realpath(self.filename)).replace("\\", "/") + "/"
        self.filename = self.filename.split("/")[-1]
        self.output_dir = self.directory + self.filename[:-4] + "_FSCT_output/"
        self.working_dir = self.directory + self.filename[:-4] + "_FSCT_output/working_directory/"

        self.filename = "working_point_cloud.las"
        self.directory = self.output_dir
        self.plot_summary = pd.read_csv(self.output_dir + "plot_summary.csv", index_col=None)
        self.plot_centre = [[float(self.plot_summary["Plot Centre X"]), float(self.plot_summary["Plot Centre Y"])]]

    def inference(self):
        test_dataset = TestingDataset(
            root_dir=self.working_dir, points_per_box=self.parameters["max_points_per_box"], device=self.device
        )

        test_loader = DataLoader(test_dataset, batch_size=self.parameters["batch_size"], shuffle=False, num_workers=0)

        model = Net(num_classes=4).to(self.device)
        if self.parameters["use_CPU_only"]:
            model.load_state_dict(
                torch.load(
                    os.path.join(get_fsct_path("model"), self.parameters["model_filename"]),
                    map_location=torch.device("cpu"),
                ),
                strict=False,
            )
        else:
            model.load_state_dict(
                torch.load(os.path.join(get_fsct_path("model"), self.parameters["model_filename"])),
                strict=False,
            )

        model.eval()
        num_boxes = test_dataset.__len__()
        with torch.no_grad():

            self.output_point_cloud = np.zeros((0, 3 + 4))
            output_list = []
            for i, data in enumerate(test_loader):
                print("\r" + str(i * self.parameters["batch_size"]) + "/" + str(num_boxes))
                data = data.to(self.device)
                out = model(data)
                out = out.permute(2, 1, 0).squeeze()
                batches = np.unique(data.batch.cpu())
                out = torch.softmax(out.cpu().detach(), axis=1)
                pos = data.pos.cpu()
                output = np.hstack((pos, out))

                for batch in batches:
                    outputb = np.asarray(output[data.batch.cpu() == batch])
                    outputb[:, :3] = outputb[:, :3] + np.asarray(data.local_shift.cpu())[3 * batch : 3 + (3 * batch)]
                    output_list.append(outputb)

            self.output_point_cloud = np.vstack(output_list)
            print("\r" + str(num_boxes) + "/" + str(num_boxes))
        del outputb, out, batches, pos, output  # clean up anything no longer needed to free RAM.
        original_point_cloud, headers = load_file(
            self.directory + self.filename, headers_of_interest=["x", "y", "z", "red", "green", "blue"]
        )
        original_point_cloud[:, :2] = original_point_cloud[:, :2] - self.plot_centre
        self.output = choose_most_confident_label(self.output_point_cloud, original_point_cloud)
        self.output = np.asarray(self.output, dtype="float64")
        self.output[:, :2] = self.output[:, :2] + self.plot_centre
        save_file(
            self.output_dir + "segmented.las",
            self.output,
            headers_of_interest=["x", "y", "z", "red", "green", "blue", "label"],
        )

        self.sem_seg_end_time = time.time()
        self.sem_seg_total_time = self.sem_seg_end_time - self.sem_seg_start_time
        self.plot_summary["Semantic Segmentation Time (s)"] = self.sem_seg_total_time
        self.plot_summary.to_csv(self.output_dir + "plot_summary.csv", index=False)
        print("Semantic segmentation took", self.sem_seg_total_time, "s")
        print("Semantic segmentation done")
        if self.parameters["delete_working_directory"]:
            shutil.rmtree(self.working_dir, ignore_errors=True)
