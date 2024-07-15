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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim, compute_depth_loss, compute_flow_loss
from gaussian_renderer.temporal_render import temporal_render
from gaussian_renderer import network_gui
import sys
from scene.temporal_scene import TemporalScene
from scene.temporal_gaussian_model import TemporalGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments.temporal import ModelParams, PipelineParams, OptimizationParams
import imutils
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from torch.utils.data import DataLoader
from scene.dataset import FourDGSdataset
from utils.flow_viz import flow_to_image


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
disable_adaptive, disable_adaopa, disable_offscale, disable_offopa, disable_morph, multiply_offopa, enable_offsh, separate_offopa, separate_offsh, regularize_opacity, enable_static, init_mode_gaussian, stop_gradient, use_skips,
new_deform, shrink_lr, use_nte, use_SE, anneal_band, anneal_band_time, mult_quaternion, rotate_sh, dynamic_sep,
use_ResFields, ewa_prune):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = TemporalGaussianModel(dataset.sh_degree, enable_offsh=enable_offsh, separate_offopa=separate_offopa, separate_offsh=separate_offsh, enable_static=enable_static, init_mode_gaussian=init_mode_gaussian,
    stop_gradient=stop_gradient, use_skips=use_skips,
    new_deform=new_deform, shrink_lr=shrink_lr, use_nte=use_nte, use_SE=use_SE,
    anneal_band=anneal_band, anneal_band_time=anneal_band_time, anneal_band_steps=dataset.anneal_band_steps,
    mult_quaternion=mult_quaternion, rotate_sh=rotate_sh,
    posbase_pe=dataset.posbase_pe, timebase_pe=dataset.timebase_pe,
        defor_depth=dataset.defor_depth, net_width=dataset.net_width,
        dynamic_sep=dynamic_sep, 
        use_ResFields=use_ResFields, ResField_mode=dataset.ResField_mode,
        capacity=dataset.capacity, ewa_prune=ewa_prune)
    scene = TemporalScene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    #dataset.white_background = white_background
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    need_abandon = False
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = temporal_render(custom_cam, gaussians, pipe, background, scaling_modifer, disable_offscale=disable_offscale, disable_offopa=disable_offopa, disable_morph=disable_morph, multiply_offopa=multiply_offopa,
                    anneal_band_iteration=iteration if (anneal_band or anneal_band_time) else None)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()


        # Pick a random Camera
        #if not viewpoint_stack:
        #    viewpoint_stack = scene.getTrainCameras().copy()
        #viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # logic to load dataset
        # if fourdgsdatset, means the very beginning of training
        #   means need the 
        # else means either very beginning or finished a loop
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras() 
            if isinstance(viewpoint_stack, FourDGSdataset):
                # at the very beginning, use a coarse filter
                viewpoint_stack.reset_kernel_size(args.kernel_size[0])
                batch_size = 1
                viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=0,collate_fn=list)
                loader = iter(viewpoint_stack_loader)
            else:
                viewpoint_stack = viewpoint_stack.copy()
        
        #assert False, [args.fine_iter_start, args.kernel_size]
        # if fourdgsdataset (hypernerf/nerfies/dycheck) and reaches the iteration of fine stage starts
        if (iteration in args.fine_iter_start) and (viewpoint_stack is not None) and isinstance(viewpoint_stack, FourDGSdataset):
            print("Start Fine Iteration")
            # only needs to set this once
            viewpoint_stack = scene.getTrainCameras()
            viewpoint_stack.reset_kernel_size(args.kernel_size[args.fine_iter_start.index(iteration)+1])
            batch_size = 1
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=0,collate_fn=list)
            loader = iter(viewpoint_stack_loader)

        
        # logic to iterate over dataset
        if isinstance(viewpoint_stack, FourDGSdataset):
            try:
                viewpoint_cam = next(loader)[0]
            except StopIteration:
                loader = iter(viewpoint_stack_loader)
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        render_pkg = temporal_render(viewpoint_cam, gaussians, pipe, background, disable_offscale=disable_offscale, disable_offopa=disable_offopa, disable_morph=disable_morph, multiply_offopa=multiply_offopa, disable_deform=iteration < opt.fix_until_iter,
            return_sep=((iteration >= opt.fix_until_iter) and ((opt.lambda_sep > 0) or (opt.lambda_prefer_static > 0))),
            return_depth=(opt.lambda_reg_depth > 0),
            return_flow=((iteration >= opt.fix_until_iter) and (opt.lambda_reg_flow > 0)),
            anneal_band_iteration=iteration if (anneal_band or anneal_band_time) else None,
            sf_reg=opt.lambda_sf_sm > 0 or opt.lambda_sf_st > 0,
            motion_gap=opt.motion_gap)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        motion_tensor = render_pkg["motion_tensor"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()[:3]
        if iteration < opt.l1_l2_switch:
            Ll1 = l2_loss(image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)

        # the case when no adpative policy is applied, use regualizer on loss
        #if not disable_adaptive:
        #    loss += 
        #    if iteration < opt.densify_until_iter:
            #loss += opt.lambda_opacity * torch.mean(gaussians.opacity_activation(gaussians._opacity))
        #        density_cond = iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0
        #        reset_cond =  not disable_adaopa and (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter))
        #        need_abandon = need_abandon or density_cond or reset_cond
        #if need_abandon:
        #    gaussians.optimizer.zero_grad(set_to_none = True)
        #    need_abandon = False
        #assert False, [Ll1.shape, torch.mean(gaussians._opacity).shape]
        #loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.lambda_opacity * torch.mean(gaussians.opacity_activation(gaussians._opacity))
        #print(need_abandon)
        #if need_abandon: # need this line to prevent loss shape mismatch with previous due to adaptive gaussian
        #    gaussians.optimizer.step()
        #    gaussians.optimizer.zero_grad(set_to_none = True)
        #    need_abandon = False

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) #+ opt.lambda_opacity * torch.mean(gaussians.opacity_activation(gaussians._opacity))
        #print(need_abandon)
        if regularize_opacity:
            loss += opt.lambda_opacity * torch.mean(gaussians.opacity_activation(gaussians._opacity))

            if need_abandon: # need this line to prevent loss shape mismatch with previous due to adaptive gaussian
                gaussians.optimizer.step()
                #gaussians.defor_optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                #gaussians.defor_optimizer.zero_grad()
                need_abandon = False
        
        if opt.lambda_reg_depth > 0:
            coeff_depth = opt.lambda_reg_depth * (1. - iteration/ float(opt.iterations))
            loss += coeff_depth * compute_depth_loss(render_pkg["rendered_depth"], viewpoint_cam.depth.cuda())
        if iteration > opt.fix_until_iter and opt.lambda_reg_flow > 0.:
            if viewpoint_cam.fwd_flow is not None and viewpoint_cam.bwd_flow is not None:
                # makes training three times slower!
                coeff_flow = opt.lambda_reg_flow * (1. - iteration/ float(opt.iterations))
                loss += coeff_flow * compute_flow_loss(
                    render_pkg["rendered_flow_fwd"], render_pkg["rendered_flow_bwd"],
                    viewpoint_cam.fwd_flow.cuda(), viewpoint_cam.bwd_flow.cuda(),
                    viewpoint_cam.fwd_flow_mask.cuda(), viewpoint_cam.bwd_flow_mask.cuda()
                )

        if opt.lambda_reg_canon > 0:
            assert not gaussians.separate_offsh and not gaussians.enable_offsh
            #disable_offopa=disable_offopa, disable_morph=disable_morph, multiply_offopa=multiply_offopa, disable_deform=iteration < opt.fix_until_iter,
            means3D, opacity, scales, rotations, _, _, _ = gaussians.get_deformed(0.5, disable_offscale, disable_offopa, disable_morph, multiply_offopa, anneal_band_iteration=iteration if (anneal_band or anneal_band_time) else None)
            loss += opt.lambda_reg_canon * (torch.mean(torch.square((means3D-gaussians.get_xyz) * gaussians._isstatic)) +\
                    torch.mean(torch.square((opacity-gaussians.get_opacity) * gaussians._isstatic)) +\
                    torch.mean(torch.square((scales-gaussians.get_scaling) * gaussians._isstatic)) +\
                    torch.mean(torch.square((rotations-gaussians.get_rotation) * gaussians._isstatic)))
        
        if opt.lambda_sf_sm > 0:
            loss += opt.lambda_sf_sm * render_pkg["sm_loss"]
        if opt.lambda_sf_st > 0:
            loss += opt.lambda_sf_st * render_pkg["st_loss"]    
        
        # regularizer for static branch
        if (iteration >= opt.fix_until_iter) and ((opt.lambda_sep > 0) or (opt.lambda_prefer_static > 0)):
            pout = render_pkg["rendered_sep"].clamp(1e-6, 1.-1e-6)
            pout = torch.pow(pout, opt.entropy_k)
            #assert False, entropy_last_loss.shape
            if opt.lambda_sep > 0:
                entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
                loss += opt.lambda_sep * entropy_last_loss
            if opt.lambda_prefer_static > 0:
                loss += opt.lambda_prefer_static * pout.mean()
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, temporal_render, 
                (pipe, background, disable_offscale, disable_offopa, disable_morph, multiply_offopa, iteration < opt.fix_until_iter, True, 
                1.0, None, False, False, False, iteration if (anneal_band or anneal_band_time) else None))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if disable_adaptive:
                pass
            elif iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                #if gaussians.ewa_prune:
                #    max_scaling = render_pkg["scaling_tensor"][visibility_filter]
                #    gaussians.max_scaling[visibility_filter] = torch.max(gaussians.max_scaling[visibility_filter], max_scaling)
                if gaussians.dynamic_sep and iteration > opt.fix_until_iter: 
                    gaussians.add_densification_stats_motion(viewspace_point_tensor, visibility_filter, motion_tensor,
                        opacity_tensor=render_pkg["opacity_tensor"], scaling_tensor=render_pkg["scaling_tensor"])
                else:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter,
                        opacity_tensor=render_pkg["opacity_tensor"], scaling_tensor=render_pkg["scaling_tensor"])

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, 
                        (iteration % opt.densification_motion_interval == 0) and (iteration>(opt.densify_motion_from_iter + opt.fix_until_iter)), opt.densify_min_motion)
                    need_abandon = True
                if disable_adaopa:
                    pass
                elif iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    #need_abandon = True

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                #gaussians.optimizer.step()
                #gaussians.defor_optimizer.step()
                #gaussians.optimizer.zero_grad(set_to_none = True)
                #gaussians.defor_optimizer.zero_grad()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : TemporalScene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        #validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
        #                      {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        if isinstance(scene.getTestCameras(), FourDGSdataset):
            validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})
        else:
            validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        #print(renderArgs)
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image, image_canon, depth = render_pkg["render"], render_pkg["rendered_canon"], render_pkg["rendered_depth"]
                    image = torch.clamp(image, 0.0, 1.0)
                    image_canon = torch.clamp(image_canon, 0.0, 1.0)
                    #print(torch.any(image != imafge_canon))
                    depth = imutils.np2png_d( [depth[0, ...].cpu().numpy()], None, colormap="jet")
                    depth = torch.from_numpy(depth).permute(2, 0, 1)
                    gt_image = torch.clamp(viewpoint.original_image[:3].to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_canon".format(viewpoint.image_name), image_canon[None], global_step=iteration)
                        image_motion = render_pkg["rendered_motion"]
                        image_motion = torch.clamp(image_motion, 0.0, 1.0)
                        tb_writer.add_images(config['name'] + "_view_{}/render_motion".format(viewpoint.image_name), image_motion[None], global_step=iteration)

                        rendered_flow_fwd = render_pkg["rendered_flow_fwd"][:2, ...].permute(1, 2, 0).cpu().numpy()
                        rendered_flow_bwd = render_pkg["rendered_flow_bwd"][:2, ...].permute(1, 2, 0).cpu().numpy()
                        try:
                            rendered_flow_fwd = flow_to_image(rendered_flow_fwd)
                            rendered_flow_bwd = flow_to_image(rendered_flow_bwd)
                            rendered_flow_fwd = torch.from_numpy(rendered_flow_fwd).permute(2, 0, 1) / 255.
                            rendered_flow_bwd = torch.from_numpy(rendered_flow_bwd).permute(2, 0, 1) / 255.
                        except:
                            rendered_flow_fwd = torch.zeros_like(image_motion)
                            rendered_flow_bwd = torch.zeros_like(image_motion)

                        tb_writer.add_images(config['name'] + '_view_{}/render_flow_fwd'.format(viewpoint.image_name), rendered_flow_fwd[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + '_view_{}/render_flow_bwd'.format(viewpoint.image_name), rendered_flow_bwd[None], global_step=iteration)


                        if render_pkg["rendered_stat"] is not None:
                            image_sep, image_dy, image_stat  = render_pkg["rendered_sep"], render_pkg["rendered_dy"], render_pkg["rendered_stat"]
                            image_dy = torch.clamp(image_dy, 0.0, 1.0)
                            image_stat = torch.clamp(image_stat, 0.0, 1.0)
                            image_sep = torch.clamp(image_sep, 0.0, 1.0)
                            
                            
                            #if scene.gaussians.enable_static:
                            tb_writer.add_images(config['name'] + "_view_{}/render_dy".format(viewpoint.image_name), image_dy[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/render_stat".format(viewpoint.image_name), image_stat[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/render_sep".format(viewpoint.image_name), image_sep[None], global_step=iteration)
                            
                     

                        #if iteration == testing_iterations[0] or iteration in fine_iter_start:
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar('dynamic_points', torch.sum(scene.gaussians._isstatic).float(), iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[100, 500, 1000, 2900, 3100, 5000, 5900, 6100, 7_000, 8900, 9100, 10000, 11900, 12100, 14900, 15100, 17900, 18100, 20000, 25000, 30_000, 35_000, 40_000, 
        45_000, 50_000, 55_000, 60_000, 70_000, 75_000, 80_000, 85_000, 90_000, 100_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--disable_adaptive", action="store_true")
    parser.add_argument("--disable_adaopa", action="store_true")
    parser.add_argument("--disable_offscale", action="store_true")
    parser.add_argument("--disable_offopa", action="store_true")
    parser.add_argument("--disable_morph", action="store_true")
    parser.add_argument("--multiply_offopa", action="store_true")
    parser.add_argument("--enable_offsh", action="store_true")
    parser.add_argument("--separate_offopa", action="store_true")
    parser.add_argument("--separate_offsh", action="store_true")
    parser.add_argument("--regularize_opacity", action="store_true")
    parser.add_argument("--enable_static", action="store_true")
    parser.add_argument("--init_mode_gaussian", action="store_true")
    parser.add_argument("--stop_gradient", action="store_true")
    parser.add_argument("--use_skips", action="store_true")
    parser.add_argument("--new_deform", action="store_true")
    parser.add_argument("--shrink_lr", action="store_true")
    parser.add_argument("--use_nte", action="store_true")
    parser.add_argument("--use_SE", action="store_true")
    parser.add_argument("--anneal_band", action="store_true")
    parser.add_argument("--anneal_band_time", action="store_true")
    parser.add_argument("--mult_quaternion", action="store_true")
    parser.add_argument("--rotate_sh", action="store_true")
    parser.add_argument("--dynamic_sep", action="store_true")
    parser.add_argument("--use_ResFields", action="store_true")
    parser.add_argument("--ewa_prune", action="store_true")


    parser.add_argument("--fine_iter_start", nargs="*", default=[], type=int)
    parser.add_argument("--kernel_size", nargs="*", default=[1.], type=float)
    
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if args.separate_offopa:
        assert not args.disable_offopa and not args.multiply_offopa, "Already using separate offopa network!"
    if args.separate_offsh:
        assert not args.enable_offsh, "Already using separate offsh network!"
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint,
    args.disable_adaptive, args.disable_adaopa, args.disable_offscale, args.disable_offopa, args.disable_morph, args.multiply_offopa, args.enable_offsh, args.separate_offopa, args.separate_offsh, args.regularize_opacity,
    args.enable_static, args.init_mode_gaussian, args.stop_gradient, args.use_skips, args.new_deform, args.shrink_lr, args.use_nte, args.use_SE, args.anneal_band, args.anneal_band_time,
    args.mult_quaternion, args.rotate_sh, args.dynamic_sep, args.use_ResFields, args.ewa_prune)

    # All done
    print("\nTraining complete.")
