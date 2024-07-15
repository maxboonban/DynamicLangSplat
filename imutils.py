from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

from pathlib import Path
import numpy as np
import struct

def read_pfm(filename):
    with Path(filename).open('rb') as pfm_file:

        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')
        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)
        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4
        
        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale

def png2np(fname, norm_input=True):
    return np.array(Image.open(fname)) / 255.

def png2torch(fname, norm_input=True):
    return (torch.tensor(np.array(Image.open(fname))) / 255.).permute(2, 0, 1)

def np2png(arr, fname, norm_input=True):
    if type(arr) == list:
        arr = np.concatenate(arr, 1)

    #plt.imsave( fname, np.uint8(norm_input * arr * 255 + (not norm_input) * arr) )
    Image.fromarray(np.uint8(norm_input * arr * 255 + (not norm_input) * arr)).save(fname)

def torch2png(arr, fname, norm_input=True):
    if type(arr) == list:
        arr = [a.permute(1, 2, 0).detach().cpu().numpy() for a in arr]
    else:
        arr = arr.permute(1, 2, 0).detach().cpu().numpy()
        
    np2png(arr, fname, norm_input)

def np2png_d(arr, fname, vmin=-1, vmax=-1, colormap="jet"):
    if type(arr) == list:
        arr = np.concatenate(arr, 1)
    
    if vmin == -1 and vmax == -1:
        vmin = np.amin(arr)
        vmax = np.amax(arr)

    #cmap = plt.cm.jet
    cmap = eval('plt.cm.%s' % colormap) #plt.cm.prism #gist_ncar #viridis
    # cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # map the normalized data to colors
    # image is now RGBA (512x512x4)
    image = cmap(norm(arr))[..., :3]

    # save the image
    if fname:
        plt.imsave(fname, image)
    else:
        return image

def torch2png_d(arr, fname, vmin=-1, vmax=-1, colormap="jet"):
    if type(arr) == list:
        return np2png_d( [a.detach().cpu().numpy() for a in arr], fname, vmin, vmax, colormap)
    else:
        return np2png_d( arr.detach().cpu().numpy(), fname, vmin, vmax, colormap)