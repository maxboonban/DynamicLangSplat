import glob
import json
import os
import random
from argparse import ArgumentParser

import einops
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

COMPONENTS = 3


def fit_dino(resolution, sequence_dir):
    dino_dir = os.path.join(sequence_dir, f"dino/{resolution}")
    output_dir = os.path.join(sequence_dir, f"dino_dim3/{resolution}")
    os.makedirs(output_dir, exist_ok=True)

    input_embeddings = sorted(os.listdir(dino_dir))

    crop_jsons = [file for file in input_embeddings if "json" in file]
    input_embeddings = [file for file in input_embeddings if "npy" in file]

    for i, file in enumerate(input_embeddings):
        features = np.load(os.path.join(dino_dir, file))
        H, W, _ = features.shape
        features = einops.rearrange(features, "h w c -> (h w) c")
        if i == 0:
            data = features
        else:
            data = np.concatenate([data, features], axis=0)
    pca = PCA(n_components=COMPONENTS)
    data_reduced = pca.fit_transform(data)

    features_reduced = einops.rearrange(
        data_reduced, "(b h1 w1) c -> b c h1 w1", b=len(input_embeddings), h1=H, w1=W
    )

    resize = nn.Upsample(scale_factor=14)
    for i, file in enumerate(input_embeddings):
        features_patch = torch.tensor(features_reduced[i]).unsqueeze(0)
        features_full = resize(features_patch).squeeze()
        with open(os.path.join(dino_dir, crop_jsons[i])) as f:
            crop_info = json.loads(f.read())
        features_cropped = features_full[
            :,
            crop_info["padding_top"] : -crop_info["padding_bottom"],
            crop_info["padding_left"] : -crop_info["padding_right"],
        ]
        features_cropped = einops.rearrange(features_cropped, "c h w -> h w c")
        np.save(os.path.join(output_dir, file), features_cropped.cpu().numpy())


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)

    parser = ArgumentParser(description="Feature Extraction parameters")
    parser.add_argument("-s", "--sequence", type=str, required=True)
    parser.add_argument("-r", "--resolution", type=int, default=2)
    parser.add_argument("-d", "--downsample", type=int, default=-1)
    args = parser.parse_args()

    images_dir = os.path.join(args.sequence, f"rgb/{args.resolution}x")
    _, _, files = [p for p in os.walk(images_dir)][0]
    img_list = []

    fit_dino(str(args.resolution) + "x", args.sequence)
