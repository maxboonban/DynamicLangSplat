import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir):
        self.data = np.load(os.path.join(data_dir, "embeddings.npy"))

    def __getitem__(self, index):
        data = torch.tensor(self.data[index, :], dtype=torch.float)
        return data

    def __len__(self):
        return self.data.shape[0]
