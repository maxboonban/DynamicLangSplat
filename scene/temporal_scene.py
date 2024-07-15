import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.temporal_dataset_readers import temporalsceneLoadTypeCallbacks
from scene.temporal_gaussian_model import TemporalGaussianModel
from arguments import ModelParams
from utils.temporal_camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np
import torch
from scene.hyper_loader import Load_hyper_data
from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
from scene.dataset import FourDGSdataset
from scene.temporal_cameras import TemporalCamera_View

class TemporalScene:

    gaussians : TemporalGaussianModel

    def __init__(self, args : ModelParams, gaussians : TemporalGaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.white_background = args.white_background

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if "plenoptic" in args.source_path:
            scene_info = temporalsceneLoadTypeCallbacks["dynerf"](args.source_path, args.eval, load_every=args.load_every, downsample=float(args.downsample), num_pts=args.num_pts)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = temporalsceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.downsample, mode=args.mode)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = temporalsceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, downsample=args.downsample)
            if self.gaussians.use_ResFields:
                assert self.gaussians.ResField_mode == "interpolation"
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json file, assuming Nerfies data set!")
            scene_info, max_time = temporalsceneLoadTypeCallbacks["nerfies"](args.source_path, args.eval, 1./float(args.downsample), args.sample_interval, num_pts=args.num_pts, num_pts_stat=args.num_pts_stat,
            num_pts_stat_extra=args.num_pts_stat_extra, bbox_range=args.bbox_range)
            if self.gaussians.use_ResFields:
                if self.gaussians.capacity < max_time:
                    assert False, f"Max Time {max_time} exceeds ResFields capacity {self.gaussians.capacity}"
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter and not isinstance(scene_info.train_cameras, Load_hyper_data) and not isinstance(scene_info.train_cameras, Neural3D_NDC_Dataset):
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle and not isinstance(scene_info.train_cameras, Load_hyper_data) and not isinstance(scene_info.train_cameras, Neural3D_NDC_Dataset):
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            if isinstance(scene_info.train_cameras, Load_hyper_data) or isinstance(scene_info.train_cameras, Neural3D_NDC_Dataset):
                self.train_cameras[resolution_scale] = FourDGSdataset(scene_info.train_cameras, args)
            else:
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            if isinstance(scene_info.test_cameras, Load_hyper_data) or isinstance(scene_info.test_cameras, Neural3D_NDC_Dataset):
                self.test_cameras[resolution_scale] = FourDGSdataset(scene_info.test_cameras, args)
            else:
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        #assert False, self.loaded_iter
        if self.loaded_iter:
            #assert False, "disabled for pointcloud_dy"
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, scene_info.point_cloud_dy)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    # adopted from K-Planes
    def getOrbitCameras(self, n_frames=120):
        train_cameras = self.getTrainCameras()
        test_cameras = self.getTestCameras()
        #origins = []
        #for camera in train_cameras + test_cameras:
        #    origins.append(camera.camera_center)

        origins = [-torch.from_numpy(camera.T) for camera in train_cameras + test_cameras]
        origins = torch.stack(origins, dim=0).cpu()
        origins[:, :2] *= 1.*1e8
        origins[:, 3:] *= 1e-2
        #assert False, train_cameras[0].T
        #assert False, origins.shape
        #origins[:, 2] *= -1
        #assert False, origins.shape
        radius = torch.sqrt(torch.mean(torch.sum(origins ** 2, dim=-1)))
        #assert False, origins[:10, 2]
        sin_phi = torch.mean(origins[:, 2], dim=0) / radius
        #assert False, [radius, sin_phi, origins]
        cos_phi = torch.sqrt(1 - sin_phi ** 2)
        render_poses = []

        up = torch.tensor([0., 0., 1.])
        for theta in np.linspace(3*np.pi, -3. * np.pi, n_frames, endpoint=False):
            camorigin = 0.5* radius * torch.tensor(
                [cos_phi * np.cos(theta), cos_phi * np.sin(theta), -sin_phi])
            #print(camorigin)
            render_poses.append(viewmatrix(camorigin, up, camorigin))

        #render_poses = torch.stack(render_poses, dim=0)
        #assert False, render_poses.shape
        # now each of these render_poses is equivalent to 'transform_matrix' values
        
        orbit_infos = []
        times = np.linspace(0., 1., len(render_poses))
        fovy = train_cameras[0].FoVy
        fovx = train_cameras[0].FoVx
        width = train_cameras[0].image_width
        height = train_cameras[0].image_height
        for idx, matrix in enumerate(render_poses):
            #assert False, matrix.shape
            matrix = np.concatenate([matrix, [[0., 0., 0., 1]]], axis=0)
            #assert False, matrix
            matrix = np.linalg.inv(matrix)
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            orbit_infos.append(
                TemporalCamera_View(colmap_id=idx, R=R, T=T, FoVx=fovx, FoVy=fovy, uid=idx, time=float(times[idx]), image_height=height, image_width=width)

            )

        
        for idx, matrix in enumerate(render_poses):
            #assert False, matrix.shape
            matrix = np.concatenate([render_poses[0], [[0., 0., 0., 1]]], axis=0)
            #assert False, matrix
            matrix = np.linalg.inv(matrix)
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            orbit_infos.append(
                TemporalCamera_View(colmap_id=idx, R=R, T=T, FoVx=fovx, FoVy=fovy, uid=idx, time=float(times[idx]), image_height=height, image_width=width)

            )
        
        return orbit_infos

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    return np.stack([-vec0, vec1, vec2, pos], axis=1)