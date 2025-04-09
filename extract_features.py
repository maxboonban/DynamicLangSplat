import os
import random
from argparse import ArgumentParser
from math import ceil, floor

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from tqdm import tqdm

DINO_FEATURES = 768
CLIP_FEATURES = 256

dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
dino.cuda()
dino.eval()


def extract_features(sequence_dir, resolution):
    images_dir = os.path.join(sequence_dir, f"rgb/{resolution}")
    mask_dir = os.path.join(sequence_dir, f"resized_mask/{resolution}")
    dino_dir = os.path.join(sequence_dir, f"dino/{resolution}")
    clip_dir = os.path.join(sequence_dir, f"clip/{resolution}")
    os.makedirs(dino_dir, exist_ok=True)
    os.makedirs(clip_dir, exist_ok=True)
    _, dirs, _ = [p for p in os.walk(mask_dir)][0]
    _, _, files = [p for p in os.walk(images_dir)][0]
    most_visible = {}
    progress_bar = tqdm(range(0, len(files)), desc="Extracting DINO features")

    for file in files:
        most_visible[file] = {dir: [-1, ""] for dir in dirs}
        image_path = os.path.join(images_dir, file)
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGB"))
        H, W, _ = im_data.shape
        im_tensor = torch.Tensor(im_data).unsqueeze(0).cuda()
        im_tensor = rearrange(im_tensor, "b h w c -> b c h w")
        H_14, W_14 = ceil(H / 14) * 14, ceil(W / 14) * 14
        padding_top, padding_bottom = ceil((H_14 - H) / 2), floor((H_14 - H) / 2)
        padding_left, padding_right = ceil((W_14 - W) / 2), floor((W_14 - W) / 2)
        assert (
            H + padding_top + padding_bottom == H_14
            and W + padding_left + padding_right == W_14
        )
        patch_padding = nn.ReflectionPad2d(
            (padding_left, padding_right, padding_top, padding_bottom)
        )

        with torch.no_grad():
            dino_dict = dino.forward_features(patch_padding(im_tensor))
        dino_features = dino_dict["x_norm_patchtokens"]
        dino_features = rearrange(dino_features, "b s c -> b c s")
        dino_features = torch.reshape(
            dino_features, (1, DINO_FEATURES, H_14 // 14, W_14 // 14)
        )
        resize = nn.Upsample(scale_factor=14)
        dino_features = resize(dino_features)
        dino_features = dino_features[
            :, :, padding_top : H + padding_top, padding_left : W + padding_left
        ]
        assert dino_features.shape[2:] == im_tensor.shape[2:]
        dino_features = rearrange(dino_features, "b c h w -> b h w c")
        np.save(os.path.join(dino_dir, file), dino_features.squeeze(0).cpu().numpy())
        for dir in dirs:
            mask_path = os.path.join(mask_dir, dir, file + ".png")
            mask = Image.open(mask_path)

            mask_data = np.array(mask.convert("RGB"))
            mask_data //= max(1, mask_data.max())
            if mask_data.sum() > most_visible[file][dir][0]:
                most_visible[file][dir] = [mask_data.sum(), mask_path]
        progress_bar.update(1)
    progress_bar.close()

    for file in files:
        image_path = os.path.join(images_dir, file)
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGB"))
        H, W, _ = im_data.shape
        clip_embeddings = None
        for dir in dirs:
            mask_path = os.path.join(mask_dir, dir, file + ".png")
            mask = Image.open(mask_path)

            mask_data = np.array(mask.convert("RGB"))
            mask_data //= max(1, mask_data.max())
            masked_image = mask_data * image
            clip_embeddings_partial = clip(masked_image)
            clip_embeddings += clip_embeddings_partial


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
    args = parser.parse_args()
    extract_features(args.sequence, str(args.resolution) + "x")
