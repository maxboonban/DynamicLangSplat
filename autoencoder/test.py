import argparse
import os

import cv2
import numpy as np
import torch
from model import Autoencoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("-r", "--resolution", type=int, default=2)
    parser.add_argument("-d", "--downsample", type=int, default=-1)
    parser.add_argument(
        "--encoder_dims",
        nargs="+",
        type=int,
        default=[256, 128, 64, 32, 3],
    )
    parser.add_argument(
        "--decoder_dims",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 256, 512],
    )
    args = parser.parse_args()

    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    checkpoint_path = f"ckpt/{dataset_name}/{args.resolution}x"
    ckpt_path = os.path.join(checkpoint_path, "best_ckpt.pth")

    data_dir = os.path.join(dataset_path, f"clip/{args.resolution}x")
    mask_dir = os.path.join(dataset_path, f"mask/{args.resolution}x")
    rgb_dir = os.path.join(dataset_path, f"rgb/{args.resolution}x")
    output_dir = os.path.join(dataset_path, f"clip_dim3/{args.resolution}x")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = torch.load(ckpt_path)

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()

    data = torch.tensor(
        np.load(os.path.join(data_dir, "embeddings.npy")), dtype=torch.float32
    ).to("cuda:0")

    outputs = model.encode(data).detach().to("cpu").numpy()

    # Create a mapping from object names to indices
    object_to_idx = {name: idx for idx, name in enumerate(sorted(os.listdir(mask_dir)))}

    os.makedirs(output_dir, exist_ok=True)

    # copy the segmentation map
    for filename in os.listdir(rgb_dir):
        clip_composite = None
        print(mask_dir)
        for mask_file in os.listdir(mask_dir):
            # Use the mask file directly
            mask_path = os.path.join(mask_dir, mask_file)
            print("This is the mask_path: ", mask_path)
            print("This is the mask file: ", mask_file)
            image = cv2.imread(mask_path)
            
            if image is None:
                continue

            orig_w, orig_h = image.shape[1], image.shape[0]
            if args.downsample == -1:
                if orig_h > 1080:
                    if not WARNED:
                        print(
                            "[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                        )
                        WARNED = True
                    global_down = orig_h / 1080
                else:
                    global_down = 1
            else:
                global_down = orig_w / args.downsample

            scale = float(global_down)
            resolution = (int(orig_w / scale), int(orig_h / scale))

            image = cv2.resize(image, resolution)
            image = np.float16(image)
            image /= 255
            image *= outputs[object_to_idx[mask_file]]

            if clip_composite is None:
                clip_composite = image
            else:
                clip_composite += image
        np.save(os.path.join(output_dir, filename), clip_composite)
