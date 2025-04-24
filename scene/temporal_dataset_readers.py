#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import copy
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from plyfile import PlyData, PlyElement
from scene.colmap_loader import (qvec2rotmat, read_extrinsics_binary,
                                 read_extrinsics_text, read_intrinsics_binary,
                                 read_intrinsics_text, read_points3D_binary,
                                 read_points3D_text)
from scene.gaussian_model import BasicPointCloud
from scene.hyper_loader import Load_hyper_data, format_hyper_data
from tqdm import tqdm
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.sh_utils import SH2RGB


class TemporalCameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    # dino_features: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time:float
    depth: Optional[np.array] = None



class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    point_cloud_dy: Optional[BasicPointCloud] = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, downsample):
    #assert False, "Not Implemented for temporal yet!"
    # TODO: read dino features
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height // downsample
        width = intr.width // downsample

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0] // downsample
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0] //downsample
            focal_length_y = intr.params[1] //downsample
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = image.resize((width, height), Image.LANCZOS)

        time = int(image_name) / (num_frames - 1)

        cam_info = TemporalCameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, time=time)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readNvidiaCameras(cam_extrinsics, cam_intrinsics, images_folder, downsample):
    #assert False, "Not Implemented for temporal yet!"
    train_cam_infos, test_cam_infos = [], []
    num_frames = len(cam_extrinsics)
    assert num_frames == 24
    
    # first get a mapping from camera id to extrinsics 
    #assert False, "start here!"
    cam_dict = {}
    for idx, key in enumerate(cam_extrinsics): #24 cameras, shuffled!!!
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()
        
        extr = cam_extrinsics[key] 
        intr = cam_intrinsics[extr.camera_id]
        
        height = intr.height // downsample
        width = intr.width // downsample

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0] // downsample
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0] //downsample
            focal_length_y = intr.params[1] //downsample
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        #image_path = os.path.join(images_folder, os.path.basename(extr.name))
        #image_name = os.path.basename(image_path).split(".")[0]
        #image = Image.open(image_path)
        #image = image.resize((width, height), Image.LANCZOS)

        #time =  / (num_frames - 1)
        #assert False, extr.camera_id
        cam_dict[key-1] = {
            "uid": uid, 
            "R": R, 
            "T": T, 
            "FovY":FovY, 
            "FovX":FovX, 
            "width": width,
            "height": height,
            #"time":int(extr.name.split(".")[0])
        }
        #print(key, extr.name)
    #assert False, [cam_dict[0]["T"], cam_dict[1]["T"], cam_dict[12]["T"], cam_dict[13]["T"]]
    #assert False, list(cam_dict.keys())
    for j in range(12):
        #for j in range(12):
        ccam = cam_dict[j]
        uid = ccam["uid"]
        R = ccam["R"]
        T = ccam["T"]
        FovY = ccam["FovY"]
        FovX = ccam["FovX"]
        width = ccam["width"]
        height = ccam["height"]
        for time in range(24):
            
            image_path = os.path.join(images_folder, "%05d" % time, "cam%02d.jpg" % (j+1)) #idx: time
            
            #image_name = os.path.basename(image_path).split(".")[0]
            image_name = str(12*time + j)
            image = Image.open(image_path)
            image = image.resize((width, height), Image.LANCZOS)

            depth_path = os.path.join(images_folder, "%05d_midasdepth" % time, "cam%02d-dpt_beit_large_512.png" % (j+1))
            if os.path.exists(depth_path):
                #assert False, depth_path
                depth = cv.imread(depth_path, -1) / (2 ** 16 - 1)  
                depth = cv.resize(depth, (width, height), cv.INTER_LANCZOS4 )
                depth = depth.astype(float)
                depth = torch.from_numpy(depth.copy())
            else:
                depth = None

             
            if j == time % 12:
                ccam = cam_dict[time]
                uidd = ccam["uid"]
                RR = ccam["R"]
                TT = ccam["T"]
                FovYY = ccam["FovY"]
                FovXX = ccam["FovX"]
                #width = ccam["width"]
                #height = ccam["height"]
                cam_info = TemporalCameraInfo(uid=uidd, R=RR, T=TT, FovY=FovYY, FovX=FovXX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, time=float(time)/(num_frames-1),
                              depth=depth)
                train_cam_infos.append(cam_info)
                #print(["train", idx, key, extr.camera_id, extr.name, j, image_path])
            else:
                cam_info = TemporalCameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, time=float(time)/(num_frames-1),
                              depth=depth)
                test_cam_infos.append(cam_info)
                #print(["test", idx, key, extr.camera_id, extr.name, j, image_path])
    #assert False, len(train_cam_infos)
    sys.stdout.write('\n')
    return train_cam_infos, test_cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, downsample, llffhold=8, mode="default"):
    assert mode in ["default", "nvidia"]
    #assert False, "Not Implemented for temporal yet!"
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    if mode == "default":
        

        reading_dir = "images" if images == None else images
        cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), downsample=downsample)
        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

        if eval:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []
    elif mode == "nvidia":
        assert False, "there must be a bug somewhere!!!"
        #reading_dir = "images" if images == None else images
        train_cam_infos, test_cam_infos = readNvidiaCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, 'mv_images'), downsample=downsample)
        #cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
        #assert False, [len(train_cam_infos), len(test_cam_infos)]
        if not eval:
        #    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        #    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        #else:
            train_cam_infos = train_cam_infos + test_cam_infos
            test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", downsample=1):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            time = frame["time"] # 0 ~ 1
            
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            depth = None

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            #fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])

            image = image.resize((image.size[0]//downsample, image.size[1]//downsample), Image.LANCZOS)

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx
            cam_infos.append(TemporalCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                            time=time, depth=depth,))
         
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, downsample, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, downsample=downsample)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, downsample=downsample)
    

    nerf_normalization = getNerfppNorm(train_cam_infos+test_cam_infos)
    if not eval:
        #train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    ply_path = os.path.join(path, "points3d.ply")
    if True or not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readHyperDataInfos(datadir,eval, ratio, sample_interval, num_pts, num_pts_stat, num_pts_stat_extra, bbox_range=2.6):
    use_bg_points = False
    train_cam_infos = Load_hyper_data(datadir,ratio,use_bg_points,split ="train", eval=eval)
    test_cam_infos = Load_hyper_data(datadir,ratio,use_bg_points,split="test", eval=eval)

    train_cam, max_time = format_hyper_data(train_cam_infos,"train")
    max_time = train_cam_infos.max_time
    video_cam_infos = copy.deepcopy(test_cam_infos)
    video_cam_infos.split="video"


    if num_pts_stat == 0:
        ply_path = os.path.join(datadir, "points.npy")

        # provided point cloud, mainly background
        xyz = np.load(ply_path,allow_pickle=True)[::sample_interval]
        
        # random scatter points within the scene bounding box
        '''
        with open(f'{datadir}/scene.json', 'r') as f:
            scene_json = json.load(f)
        if "bbox" in scene_json:
            bbox = np.array(scene_json["bbox"])*0.75 #2,3, first row smallest, second row biggest
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")
            
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.concatenate([xyz, np.random.random((num_pts, 3)) * (bbox[1:]-bbox[:1]) + bbox[:1]], axis=0)
            #assert False, [bbox.shape, bbox]
        '''

        xyz -= train_cam_infos.scene_center
        xyz *= train_cam_infos.coord_scale
        xyz = xyz.astype(np.float32)
        #color_ply_path = os.path.join(datadir, "color_points.npy")
        #if os.path.exists(color_ply_path):
        #    colors = np.load(color_ply_path, allow_pickle=True)[::sample_interval] # this is already between 0 and 1.
        #    pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros((xyz.shape[0], 3)))
        #else:  
        if num_pts_stat_extra > 0:
            xyz = np.concatenate([
                xyz,
                np.random.random((num_pts_stat_extra, 3)) * bbox_range - 0.5*bbox_range],
                axis=0)
       
        
        shs = np.random.random((xyz.shape[0], 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))
    else:
        ply_path = None
        xyz = np.random.random((num_pts_stat, 3)) * bbox_range - 0.5*bbox_range
        shs = np.random.random((num_pts_stat, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts_stat, 3)))
    
    if num_pts > 0:
        xyz_dy = np.random.random((num_pts, 3))
        xyz_dy = (xyz_dy-0.5) *bbox_range
        #xyz_dy = xyz_dy *0.25
        #xyz_dy[:, 2] -= 0.5 # z-axis; minus bigger, closer to camera
        #xyz_dy[:, 1] -= 0.25 # vertical; minux bigger, box higher
        #xyz_dy[:, 0] -= 0.25 # horizontal; minux bigger, box more left
        shs_dy = np.random.random((num_pts, 3)) / 255.0
        pcd_dy = BasicPointCloud(points=xyz_dy, colors=SH2RGB(shs_dy), normals=np.zeros((num_pts, 3)))
    else:
        pcd_dy = None


    nerf_normalization = getNerfppNorm(train_cam)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           #video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           point_cloud_dy=pcd_dy
                           #maxtime=max_time
                           )

    return scene_info, max_time
def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[0], image.shape[1])
            FovY = focal2fov(dataset.focal[0], image.shape[2])
            cameras.append(TemporalCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, depth=None))

    return cameras

def format_render_poses(poses,data_infos):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(data_infos.focal[0], image.shape[2])
        FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cameras.append(TemporalCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                            time = time, depth=None))
    return cameras


def readdynerfInfo(datadir, eval, load_every, downsample, num_pts):
    # loading all the data follow hexplane format
    ply_path = os.path.join(datadir, "points3d.ply")

    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
        datadir,
        "train",
        downsample=downsample,
        time_scale=1,
        scene_bbox_min=[-2.5, -2.0, -1.0],
        scene_bbox_max=[2.5, 2.0, 1.0],
        eval_index=0,
        load_every=load_every
        )    
    test_dataset = Neural3D_NDC_Dataset(
        datadir,
        "test",
        downsample=downsample,
        time_scale=1,
        scene_bbox_min=[-2.5, -2.0, -1.0],
        scene_bbox_max=[2.5, 2.0, 1.0],
        eval_index=0,
        load_every=load_every
        )
    train_cam_infos = format_infos(train_dataset,"train")
    
    # test_cam_infos = format_infos(test_dataset,"test")
    val_cam_infos = format_render_poses(test_dataset.val_poses,test_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)
    # create pcd
    # if not os.path.exists(ply_path):
    # Since this data set has no colmap data, we start with random points
    assert False, "don't waste time here until we have colmap masked point cloud from first frame"
    if num_pts > 0:
        xyz_dy = np.random.random((num_pts, 3)) 
        xyz_dy[..., :1]= xyz_dy[..., :1]* 1. -.5
        xyz_dy[..., 1:2]= xyz_dy[..., 1:2]* 1. - .5
        xyz_dy[..., 2:]= xyz_dy[..., 2:]*3 -1.5 #[0, 1] not visible; [-0.5, 0.5] can see part, [-5, -4] can see all
        shs_dy = np.random.random((num_pts, 3)) / 255.0
        pcd_dy = BasicPointCloud(points=xyz_dy, colors=SH2RGB(shs_dy), normals=np.zeros((num_pts, 3)))
    else:
        pcd_dy = None
    
    num_pts = 10000
    print(f"Generating random point cloud ({num_pts})...")
    threshold = 3
    # xyz_max = np.array([1.5*threshold, 1.5*threshold, 1.5*threshold])
    # xyz_min = np.array([-1.5*threshold, -1.5*threshold, -3*threshold])
    xyz_max = np.array([1.5*threshold, 1.5*threshold, 1.5*threshold])
    xyz_min = np.array([-1.5*threshold, -1.5*threshold, -1.5*threshold])
    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = (np.random.random((num_pts, 3)))* (xyz_max-xyz_min) + xyz_min
    print("point cloud initialization:",xyz.max(axis=0),xyz.min(axis=0))
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        # xyz = np.load
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           #video_cameras=val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           point_cloud_dy=pcd_dy
                           #maxtime=300
                           )
    return scene_info

temporalsceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "nerfies": readHyperDataInfos, #use 4DGaussians' dataloader
    "dynerf": readdynerfInfo
}