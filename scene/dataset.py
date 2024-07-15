from torch.utils.data import Dataset
from scene.temporal_cameras import TemporalCamera as Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
import kornia

class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
    ):
        self.dataset = dataset
        self.args = args
        self.kernel_size = 1.
    def __getitem__(self, index):

        try:
            #assert False, "depth not supported"
            image, w2c, time = self.dataset[index]
            R,T = w2c
            FovX = focal2fov(self.dataset.focal[0], image.shape[2])
            FovY = focal2fov(self.dataset.focal[0], image.shape[1])
            depth = None
            fwd_flow = None 
            fwd_flow_mask = None
            bwd_flow = None 
            bwd_flow_mask = None
        except:
            caminfo = self.dataset[index]
            image = caminfo.image
            R = caminfo.R
            T = caminfo.T
            FovX = caminfo.FovX
            FovY = caminfo.FovY
            time = caminfo.time
            depth = caminfo.depth
            fwd_flow = caminfo.fwd_flow
            fwd_flow_mask = caminfo.fwd_flow_mask
            bwd_flow = caminfo.bwd_flow 
            bwd_flow_mask = caminfo.bwd_flow_mask
            frame_id = caminfo.frame_id
        #assert False, [type(image), image.shape]
        if self.kernel_size > 1.:
            assert False, "Disabled for now"
            image = image.unsqueeze(0)
            image = kornia.filters.gaussian_blur2d(image, (self.kernel_size, self.kernel_size), (self.kernel_size/2., self.kernel_size/2.))[0]
            #image = kornia.filters.bilateral_blur(image, (self.kernel_size, self.kernel_size), 0.1, (self.kernel_size/2., self.kernel_size/2.))[0]
            #print(image.shape)
            #image = kornia.filters.median_blur(image, (self.kernel_size, self.kernel_size))[0]
            #assert False, image.shape
            if depth is not None:
                depth = depth[None, None, ...]
                depth = kornia.filters.gaussian_blur2d(depth, (self.kernel_size, self.kernel_size), (self.kernel_size/2., self.kernel_size/2.))[0, 0]
            assert False, "anyway not supported for flows for now"
        return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                          image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                          depth=depth,
                          fwd_flow=fwd_flow, fwd_flow_mask=fwd_flow_mask,
                          bwd_flow=bwd_flow, bwd_flow_mask=bwd_flow_mask,
                          frame_id=frame_id)
    def __len__(self):
        
        return len(self.dataset)
    
    def reset_kernel_size(self, kernel_size):
        self.kernel_size = kernel_size