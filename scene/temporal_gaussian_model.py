import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, build_rotation
import torch.nn.functional as F
#from dqtorch import quaternion_mul as batch_quaternion_multiply_new
import scene.resfields as resfields
import math

class Sine(nn.Module):
    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

#Adopted from TiNeuVox
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_views=3, input_ch_time=9, skips=[], max_sh_degree=None, opa_only=False, sh_only=False, init_mode_gaussian=False,
        use_nte=False,
        use_SE=False,
        use_ResFields=False, composition_rank=10, capacity=10,
        mode='lookup', compression='vm',
        fuse_mode='add', coeff_ratio=1.0, ResField_timein=True):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.sh_head = None
        self.opa_only = opa_only
        self.sh_only = sh_only
        self.max_sh_degree = max_sh_degree
        assert not opa_only or max_sh_degree is None, "When opa only network, should not use sh offset"
        assert not opa_only or not sh_only, "When only using opa offset, do not bother sh offset"
        
        self.use_SE=use_SE
        self.use_ResFields=use_ResFields
        if self.use_ResFields:
            self.nl = Sine()
            # https://github.com/markomih/ResFields/blob/a3ec626a76e95cc6e3bf571b5f1b47ed8afd66fb/dyrecon/configs/dysdf/base.yaml
            # https://github.com/markomih/ResFields/blob/a3ec626a76e95cc6e3bf571b5f1b47ed8afd66fb/dyrecon/configs/dysdf/dnerf.yaml
            self.composition_rank = composition_rank 
            self.capacity = capacity
            self.mode = mode
            self.compression = compression
            self.fuse_mode = fuse_mode
            self.coeff_ratio = coeff_ratio
            self.ResField_timein = ResField_timein
            #self.resfields_layers = self.sk 
            #self.skips = [] # not compatible 
            self.init_mode_gaussian = False # not compatible

        if opa_only:
            self._time, self.opaq_head = self.create_net_opaq()
        elif sh_only:
            self._time, self.sh_head_dc, self.sh_head_extra = self.create_net_sh(max_sh_degree)
        elif max_sh_degree is None:
            if self.use_SE:
                self._time, self.w_head, self.v_head, self.scale_head, self.rot_head, self.opaq_head = self.create_net_SE()
            else:
                self._time, self.pos_head, self.scale_head, self.rot_head, self.opaq_head = self.create_net()
        else:
            if self.use_SE:
                self._time, self.w_head, self.v_head, self.scale_head, self.rot_head, self.opaq_head, self.sh_head = self.create_net_SE(max_sh_degree)
            else:
                self._time, self.pos_head, self.scale_head, self.rot_head, self.opaq_head, self.sh_head = self.create_net(max_sh_degree)
        
        self.use_nte = use_nte
        if use_nte:
            self.timenet = nn.Sequential(
                nn.Linear(input_ch_time, W), nn.ReLU(inplace=True),
                nn.Linear(W, input_ch_time))
        
        if init_mode_gaussian:
            self.apply(self._init_weights)
    
    @staticmethod
    @torch.no_grad()
    def sine_init(m):
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
    
    @staticmethod
    @torch.no_grad()
    def first_layer_sine_init(m):
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=.001)
            module.weight.data.normal_(mean=0.0, std=.001)
    def create_net(self, max_sh_degree=None):
        if self.use_ResFields:
            return self.create_net_ResFields(max_sh_degree=max_sh_degree)
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 2):
            layer = nn.Linear
            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch + self.input_ch_time
            layers += [layer(in_channels, self.W)]
        if max_sh_degree is None:
            return nn.ModuleList(layers), nn.Linear(self.W, 3), nn.Linear(self.W, 3), nn.Linear(self.W, 4), nn.Linear(self.W, 1)
        else:
            return nn.ModuleList(layers), nn.Linear(self.W, 3), nn.Linear(self.W, 3), nn.Linear(self.W, 4), nn.Linear(self.W, 1), nn.Linear(self.W, 3 * (max_sh_degree + 1) ** 2)
    def create_net_ResFields(self, max_sh_degree=None):
        
        dims = [self.input_ch + self.input_ch_time] + [self.W for _ in range(self.D-1)]
        layers = []
        #layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 1):
            _rank = self.composition_rank if i in self.skips else 0
            _capacity = self.capacity if i in self.skips else 0
            #if i in self.skips:
            #    assert False, "Not compatible with skip layers for now"
            #    in_channels += self.input_ch + self.input_ch_time
            lin = resfields.Linear(
                dims[i], dims[i + 1], 
                rank=_rank, capacity=_capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
            lin.apply(self.first_layer_sine_init if i == 0 else self.sine_init)
            
            layers += [lin]
        if max_sh_degree is None:
            return nn.ModuleList(layers), nn.Linear(self.W, 3), nn.Linear(self.W, 3), nn.Linear(self.W, 4), nn.Linear(self.W, 1)
        else:
            return nn.ModuleList(layers), nn.Linear(self.W, 3), nn.Linear(self.W, 3), nn.Linear(self.W, 4), nn.Linear(self.W, 1), nn.Linear(self.W, 3 * (max_sh_degree + 1) ** 2)
    
        pos_head = resfields.Linear(
                dims[-1], 3, 
                rank=self.composition_rank, capacity=self.capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
        pos_head.apply(self.sine_init)
            
        scale_head = resfields.Linear(
                dims[-1], 3, 
                rank=self.composition_rank, capacity=self.capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
        scale_head.apply(self.sine_init)

        rot_head = resfields.Linear(
                dims[-1], 4, 
                rank=self.composition_rank, capacity=self.capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
        rot_head.apply(self.sine_init)
        
        opaq_head = resfields.Linear(
                dims[-1], 1, 
                rank=self.composition_rank, capacity=self.capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
        opaq_head.apply(self.sine_init)

        if max_sh_degree is None:
            return nn.ModuleList(layers), pos_head, scale_head, rot_head, opaq_head
        else:
            sh_head = resfields.Linear(
                dims[-1], 3 * (max_sh_degree + 1) ** 2, 
                rank=self.composition_rank, capacity=self.capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
            sh_head.apply(self.sine_init)
            return nn.ModuleList(layers), pos_head, scale_head, rot_head, opaq_head, sh_head
    
    
    def create_net_SE(self, max_sh_degree=None):
        if self.use_ResFields:
            return self.create_net_SE_ResFields(max_sh_degree=max_sh_degree)
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 2):
            layer = nn.Linear
            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch + self.input_ch_time
            layers += [layer(in_channels, self.W)]
        if max_sh_degree is None:
            return nn.ModuleList(layers), nn.Linear(self.W, 3), nn.Linear(self.W, 3), nn.Linear(self.W, 3), nn.Linear(self.W, 4), nn.Linear(self.W, 1)
        else:
            return nn.ModuleList(layers), nn.Linear(self.W, 3), nn.Linear(self.W, 3), nn.Linear(self.W, 3), nn.Linear(self.W, 4), nn.Linear(self.W, 1), nn.Linear(self.W, 3 * (max_sh_degree + 1) ** 2)

    def create_net_SE_ResFields(self, max_sh_degree=None):
        
        dims = [self.input_ch + self.input_ch_time] + [self.W for _ in range(self.D-1)]
        layers = []
        #layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 1):
            _rank = self.composition_rank if (i in self.skips) else 0
            _capacity = self.capacity if (i in self.skips) else 0
            #if i in self.skips:
            #    assert False, "Not compatible with skip layers for now"
            #    in_channels += self.input_ch + self.input_ch_time
            lin = resfields.Linear(
                dims[i], dims[i + 1], 
                rank=_rank, capacity=_capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
            lin.apply(self.first_layer_sine_init if i == 0 else self.sine_init)
            
            layers += [lin]
        if max_sh_degree is None:
            return nn.ModuleList(layers), nn.Linear(self.W, 3), nn.Linear(self.W, 3), nn.Linear(self.W, 3), nn.Linear(self.W, 4), nn.Linear(self.W, 1)
        else:
            return nn.ModuleList(layers), nn.Linear(self.W, 3), nn.Linear(self.W, 3), nn.Linear(self.W, 3), nn.Linear(self.W, 4), nn.Linear(self.W, 1), nn.Linear(self.W, 3 * (max_sh_degree + 1) ** 2)

        w_head = resfields.Linear(
                dims[-1], 3, 
                rank=self.composition_rank, capacity=self.capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
        w_head.apply(self.sine_init)
        v_head = resfields.Linear(
                dims[-1], 3, 
                rank=self.composition_rank, capacity=self.capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
        v_head.apply(self.sine_init)

        scale_head = resfields.Linear(
                dims[-1], 3, 
                rank=self.composition_rank, capacity=self.capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
        scale_head.apply(self.sine_init)

        rot_head = resfields.Linear(
                dims[-1], 4, 
                rank=self.composition_rank, capacity=self.capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
        rot_head.apply(self.sine_init)
        
        opaq_head = resfields.Linear(
                dims[-1], 1, 
                rank=self.composition_rank, capacity=self.capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
        opaq_head.apply(self.sine_init)

        if max_sh_degree is None:
            return nn.ModuleList(layers), w_head, v_head, scale_head, rot_head, opaq_head
        else:
            sh_head = resfields.Linear(
                dims[-1], 3 * (max_sh_degree + 1) ** 2, 
                rank=self.composition_rank, capacity=self.capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
            sh_head.apply(self.sine_init)
            return nn.ModuleList(layers), w_head, v_head, scale_head, rot_head, opaq_head, sh_head
    

    def create_net_opaq(self):
        if self.use_ResFields:
            return self.create_net_opaq_ResFields()
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 2):
            layer = nn.Linear
            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch + self.input_ch_time
            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 1)
    
    def create_net_opaq_ResFields(self, max_sh_degree=None):
        
        dims = [self.input_ch + self.input_ch_time] + [self.W for _ in range(self.D-1)]
        layers = []
        #layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 1):
            _rank = self.composition_rank if (i in self.skips) else 0
            _capacity = self.capacity if (i in self.skips) else 0
            #if i in self.skips:
            #    assert False, "Not compatible with skip layers for now"
            #    in_channels += self.input_ch + self.input_ch_time
            lin = resfields.Linear(
                dims[i], dims[i + 1], 
                rank=_rank, capacity=_capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
            lin.apply(self.first_layer_sine_init if i == 0 else self.sine_init)
            
            layers += [lin]
        return nn.ModuleList(layers), nn.Linear(self.W, 1)
        opaq_head = resfields.Linear(
                dims[-1], 1, 
                rank=self.composition_rank, capacity=self.capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
        opaq_head.apply(self.sine_init)

        return nn.ModuleList(layers), opaq_head
    
    
    
    def create_net_sh(self, max_sh_degree):
        if self.use_ResFields:
            return self.create_net_sh_ResFields(max_sh_degree=max_sh_degree)
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 2):
            layer = nn.Linear
            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch + self.input_ch_time
            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 3 * 1), nn.Linear(self.W, 3 * ((max_sh_degree + 1) ** 2-1))
        #else:
        #    return nn.ModuleList(layers), nn.Linear(self.W, 3), nn.Linear(self.W, 3), nn.Linear(self.W, 4), nn.Linear(self.W, 1), nn.Linear(self.W, 3 * (max_sh_degree + 1) ** 2)

    def create_net_sh_ResFields(self, max_sh_degree=None):
        
        dims = [self.input_ch + self.input_ch_time] + [self.W for _ in range(self.D-1)]
        layers = []
        #layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 1):
            _rank = self.composition_rank if (i in self.skips) else 0
            _capacity = self.capacity if (i in self.skips) else 0
            #if i in self.skips:
            #    assert False, "Not compatible with skip layers for now"
            #    in_channels += self.input_ch + self.input_ch_time
            lin = resfields.Linear(
                dims[i], dims[i + 1], 
                rank=_rank, capacity=_capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
            lin.apply(self.first_layer_sine_init if i == 0 else self.sine_init)
            
            layers += [lin]
        return nn.ModuleList(layers), nn.Linear(self.W, 3 * 1), nn.Linear(self.W, 3 * ((max_sh_degree + 1) ** 2-1))
        
        sh_head_dc = resfields.Linear(
                dims[-1], 3, 
                rank=self.composition_rank, capacity=self.capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
        sh_head_dc.apply(self.sine_init)

        sh_head_extra = resfields.Linear(
                dims[-1], 3 * ((max_sh_degree + 1) ** 2-1), 
                rank=self.composition_rank, capacity=self.capacity, 
                mode=self.mode, compression=self.compression, 
                fuse_mode=self.fuse_mode, coeff_ratio=self.coeff_ratio)
        sh_head_extra.apply(self.sine_init)

        

        return nn.ModuleList(layers), sh_head_dc, sh_head_extra

    def query_time(self, new_pts, t, net, pos_head, scale_head, rot_head, opaq_head):
        if self.use_ResFields:
            return self.query_time_ResFields(new_pts, t, net, pos_head, scale_head, rot_head, opaq_head)
        
        if self.use_nte:
            t = self.timenet(t)
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            #print(i, net[i].weight.device, net[i].bias.device, h.device)
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips and i != len(net)-1 :
                h = torch.cat([new_pts, t, h], -1)
        return pos_head(h), scale_head(h), rot_head(h), opaq_head(h)
    
    def query_time_ResFields(self, new_pts, t, net, pos_head, scale_head, rot_head, opaq_head):
        t, t_embed, frame_id = t
        t = 2. * t - 1.
        h = torch.cat([new_pts, t_embed], dim=1)
        h = h[:, None, :] # have to [B, ?] -> [B, 1, ?]
        for i, lin in enumerate(net):
            #print(h)
            h = lin(h, input_time=t, frame_id=frame_id)
            #print(h)
            h = self.nl(h)
        #print(h)
        h = h[:, 0, :]
        return pos_head(h), scale_head(h), rot_head(h), opaq_head(h)
    
        #return pos_head(h, input_time=t, frame_id=None)[:, 0, :],\
        #    scale_head(h, input_time=t, frame_id=None)[:, 0, :],\
        #    rot_head(h, input_time=t, frame_id=None)[:, 0, :],\
        #    opaq_head(h, input_time=t, frame_id=None)[:, 0, :]
    
    def query_time_SE(self, new_pts, t, net, w_head, v_head, scale_head, rot_head, opaq_head):
        if self.use_ResFields:
            return self.query_time_SE_ResFields(new_pts, t, net, w_head, v_head, scale_head, rot_head, opaq_head)
        if self.use_nte:
            t = self.timenet(t)
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            #print(i, net[i].weight.device, net[i].bias.device, h.device)
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips and i != len(net)-1 :
                h = torch.cat([new_pts, t, h], -1)
        return w_head(h), v_head(h), scale_head(h), rot_head(h), opaq_head(h)
    def query_time_SE_ResFields(self, new_pts, t, net, w_head, v_head, scale_head, rot_head, opaq_head):
        t, t_embed, frame_id = t
        t = 2. * t - 1.
        h = torch.cat([new_pts, t_embed], dim=1)
        h = h[:, None, :]
        for i, lin in enumerate(net):
            h = self.nl(lin(h, input_time=t, frame_id=frame_id))
        h = h[:, 0, :]
        return w_head(h), v_head(h), scale_head(h), rot_head(h), opaq_head(h)
    
        #return w_head(h, input_time=t, frame_id=None)[:, 0, :],\
        #    v_head(h, input_time=t, frame_id=None)[:, 0, :],\
        #    scale_head(h, input_time=t, frame_id=None)[:, 0, :],\
        #    rot_head(h, input_time=t, frame_id=None)[:, 0, :],\
        #    opaq_head(h, input_time=t, frame_id=None)[:, 0, :]
    
    
    def query_time_sh(self, new_pts, t, net, pos_head, scale_head, rot_head, opaq_head, sh_head):
        if self.use_ResFields:
            return self.query_time_sh_ResFields(new_pts, t, net, pos_head, scale_head, rot_head, opaq_head, sh_head)
        if self.use_nte:
            t = self.timenet(t)
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            #print(i, net[i].weight.device, net[i].bias.device, h.device)
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips and i != len(net)-1 :
                h = torch.cat([new_pts, t, h], -1)
        return pos_head(h), scale_head(h), rot_head(h), opaq_head(h), sh_head(h).view(h.shape[0], (self.max_sh_degree + 1) ** 2, 3)
    def query_time_sh_ResFields(self, new_pts, t, net, pos_head, scale_head, rot_head, opaq_head, sh_head):
        t, t_embed, frame_id = t
        t = 2. * t - 1.
        h = torch.cat([new_pts, t_embed], dim=1)
        h = h[:, None, :]
        for i, lin in enumerate(net):
            h = self.nl(lin(h, input_time=t, frame_id=frame_id))
        h = h[:, 0, :]
        return pos_head(h), scale_head(h), rot_head(h), opaq_head(h), sh_head(h).view(h.shape[0], (self.max_sh_degree + 1) ** 2, 3)

        #return pos_head(h, input_time=t, frame_id=None)[:, 0, :],\
        #    scale_head(h, input_time=t, frame_id=None)[:, 0, :],\
        #    rot_head(h, input_time=t, frame_id=None)[:, 0, :],\
        #    opaq_head(h, input_time=t, frame_id=None)[:, 0, :],\
        #    sh_head(h, input_time=t, frame_id=None)[:, 0, :]
    
    def query_time_sh_SE(self, new_pts, t, net, w_head, v_head, scale_head, rot_head, opaq_head, sh_head):
        if self.use_ResFields:
            return self.query_time_sh_SE_ResFields(new_pts, t, net, w_head, v_head, scale_head, rot_head, opaq_head, sh_head)
        if self.use_nte:
            t = self.timenet(t)
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            #print(i, net[i].weight.device, net[i].bias.device, h.device)
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips and i != len(net)-1 :
                h = torch.cat([new_pts, t, h], -1)
        return w_head(h), v_head(h), scale_head(h), rot_head(h), opaq_head(h), sh_head(h).view(h.shape[0], (self.max_sh_degree + 1) ** 2, 3)
    
    def query_time_sh_SE_ResFields(self, new_pts, t, net, w_head, v_head, scale_head, rot_head, opaq_head, sh_head):
        t, t_embed, frame_id = t
        t = 2. * t - 1.
        h = torch.cat([new_pts, t_embed], dim=1)
        h = h[:, None, :]
        for i, lin in enumerate(net):
            h = self.nl(lin(h, input_time=t, frame_id=frame_id))
        h = h[:, 0, :]
        return w_head(h), v_head(h), scale_head(h), rot_head(h), opaq_head(h), sh_head(h).view(h.shape[0], (self.max_sh_degree + 1) ** 2, 3)

        #return w_head(h, input_time=t, frame_id=None)[:, 0, :],\
        #    v_head(h, input_time=t, frame_id=None)[:, 0, :],\
        #    scale_head(h, input_time=t, frame_id=None)[:, 0, :],\
        #    rot_head(h, input_time=t, frame_id=None)[:, 0, :],\
        #    opaq_head(h, input_time=t, frame_id=None)[:, 0, :],\
        #    sh_head(h, input_time=t, frame_id=None)[:, 0, :]
        

    def query_time_opaq(self, new_pts, t, net, opaq_head):
        if self.use_ResFields:
            return self.query_time_opaq_ResFields(new_pts, t, net, opaq_head)
        if self.use_nte:
            t = self.timenet(t)
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            #print(i, net[i].weight.device, net[i].bias.device, h.device)
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips and i != len(net)-1 :
                h = torch.cat([new_pts, t, h], -1)
        return opaq_head(h)
    def query_time_opaq_ResFields(self, new_pts, t, net, opaq_head):
        t, t_embed, frame_id = t
        t = 2. * t - 1.
        h = torch.cat([new_pts, t_embed], dim=1)
        h = h[:, None, :]
        for i, lin in enumerate(net):
            h = self.nl(lin(h, input_time=t, frame_id=frame_id))
        h = h[:, 0, :]
        return opaq_head(h)
        #return opaq_head(h, input_time=t, frame_id=None)[:, 0, :]

    def query_time_sh_only(self, new_pts, t, net, sh_head_dc, sh_head_extra):
        if self.use_ResFields:
            return self.query_time_sh_only_ResFields(new_pts, t, net, sh_head_dc, sh_head_extra)
        if self.use_nte:
            t = self.timenet(t)
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            #print(i, net[i].weight.device, net[i].bias.device, h.device)
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips and i != len(net)-1 :
                h = torch.cat([new_pts, t, h], -1)
        return sh_head_dc(h).view(h.shape[0], 1, 3), sh_head_extra(h).view(h.shape[0], (self.max_sh_degree + 1) ** 2-1, 3)
    
    def query_time_sh_only_ResFields(self, new_pts, t, net, sh_head_dc, sh_head_extra):
        t, t_embed, frame_id = t
        t = 2. * t - 1.
        h = torch.cat([new_pts, t_embed], dim=1)
        h = h[:, None, :]
        for i, lin in enumerate(net):
            h = self.nl(lin(h, input_time=t, frame_id=frame_id))
        h = h[:, 0, :]
        return sh_head_dc(h).view(h.shape[0], 1, 3), sh_head_extra(h).view(h.shape[0], (self.max_sh_degree + 1) ** 2-1, 3)
        #
        #return sh_head_dc(h, input_time=t, frame_id=None)[:, 0, :],\
        #    sh_head_extra(h, input_time=t, frame_id=None)[:, 0, :]

    def forward(self, input_pts, ts):
        if self.use_SE:
            return self.forward_SE(input_pts, ts)
        if self.opa_only:
            return self.query_time_opaq(input_pts, ts, self._time, self.opaq_head)
        elif self.sh_only:
            return self.query_time_sh_only(input_pts, ts, self._time, self.sh_head_dc, self.sh_head_extra)
        elif self.sh_head is None:
            return self.query_time(input_pts, ts, self._time, self.pos_head, self.scale_head, self.rot_head, self.opaq_head)
        else:
            return self.query_time_sh(input_pts, ts, self._time, self.pos_head, self.scale_head, self.rot_head, self.opaq_head, self.sh_head)
        #input_pts_orig = input_pts[:, :3]
        #out=input_pts_orig + dx
        #return dpos, dscale, drot, dopaq
    def forward_SE(self, input_pts, ts):
        if self.opa_only:
            return self.query_time_opaq(input_pts, ts, self._time, self.opaq_head)
        elif self.sh_only:
            return self.query_time_sh_only(input_pts, ts, self._time, self.sh_head_dc, self.sh_head_extra)
        elif self.sh_head is None:
            return self.query_time_SE(input_pts, ts, self._time, self.w_head, self.v_head, self.scale_head, self.rot_head, self.opaq_head)
        else:
            return self.query_time_sh_SE(input_pts, ts, self._time, self.w_head, self.v_head, self.scale_head, self.rot_head, self.opaq_head, self.sh_head)
        
# input_data if is position, Bx3
# input_data if is time. Bx1
def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    #assert False, input_data_emb.shape
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb

# Adopted from Nerfies
def cosine_easing_window(min_freq_log2, max_freq_log2, num_bands, alpha):
    """Eases in each frequency one by one with a cosine.

    This is equivalent to taking a Tukey window and sliding it to the right
    along the frequency spectrum.

    Args:
      min_freq_log2: the lower frequency band.
      max_freq_log2: the upper frequency band.
      num_bands: the number of frequencies.
      alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.

    Returns:
      A 1-d with num_sample elements containing the window.
    """
    if min_freq_log2 is None:
        min_freq_log2 = 0.
    if max_freq_log2 is None:
        max_freq_log2 = num_bands - 1.0
    bands = torch.linspace(min_freq_log2, max_freq_log2, num_bands).cuda()
    x = torch.clamp(alpha - bands, min=0.0, max=1.0)
    return 0.5 * (1 + torch.cos(torch.pi * x + torch.pi))

class CosineEasingSchedule:
    """Schedule that eases slowsly using a cosine."""
    def __init__(self, initial_value, final_value, num_steps):
        super().__init__()
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_steps = num_steps
    def get(self, step):
        """Get the value for the given step."""
        alpha = torch.minimum(step / self.num_steps, 1.0)
        scale = self.final_value - self.initial_value
        x = min(max(alpha, 0.0), 1.0)
        return (self.initial_value
            + scale * 0.5 * (1 + torch.cos(torch.pi * x + torch.pi)))
class LinearSchedule:
    """Linearly scaled scheduler."""
    def __init__(self, initial_value, final_value, num_steps):
        super().__init__()
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_steps = num_steps
    
    def get(self, step):
        """Get the value for the given step."""
        if self.num_steps == 0:
            return torch.Tensor([step]).to("cuda").float().repeat(int(self.final_value))
            #return torch.full_like(step, self.final_value, dtype=torch.float32)
        step = torch.Tensor([step]).to("cuda").float().repeat(int(self.final_value))
        alpha = torch.clamp(step / self.num_steps, min=-1.0, max=1.0)
        return (1.0 - alpha) * self.initial_value + alpha * self.final_value

# adopted from SHRotation: http://filmicworlds.com/blog/simple-and-fast-spherical-harmonic-rotation/
# and https://zhuanlan.zhihu.com/p/51267461
# batch size N
# have middle dim with size 3 because 3-channel color
# Input
#   q: Nx4 -> quaternion of each Gaussian
#   features: Nx3x1/4/9 -> SH of each Gaussian
# Output
#   feat_out: Nx3x1/4/9 -> rotated SH of each Gaussian
def batch_SH_rotate_old(q, features):
    #assert False, features.shape
    features = features.permute(0, 2, 1)
    if features.shape[-1] > 9:
        features_left = features[..., 9:]
    else:
        features_left = None
        assert features.shape[-1] in [1, 4, 9], "At max support SH order of 3"
    
    if features.shape[-1] == 1:
        return features.permute(0, 2, 1) # order 1 is rotation-invariant; don't need to change anything 

    # get rotation matrix M:Nx3x3 given quaternion x
    # https://www.songho.ca/opengl/gl_quaternion.html
    # Nx1
    s, x, y, z = q[:, :1], q[:, 1:2], q[:, 2:3], q[:, 3:4] 
    m00_ = 1 - 2 * y**2 - 2 * z**2
    m01_ = 2 * x * y - 2 * s * z
    m02_ = 2 * x * z + 2 * s * y
    m10_ = 2 * x * y + 2 * s * z
    m11_ = 1 - 2 * x**2 - 2 * z**2
    m12_ = 2 * y * z - 2 * s * x
    m20_ = 2 * x * z - 2 * s * y
    m21_ = 2 * y * z + 2 * s * x
    m22_ = 1 - 2 * x**2 - 2 * y**2
    M = torch.stack([
        torch.cat([m00_, m01_, m02_], dim=-1), #Nx3
        torch.cat([m10_, m11_, m12_], dim=-1), #Nx3
        torch.cat([m20_, m21_, m22_], dim=-1), #Nx3
    ], dim=1) #Nx3x3
    M = M.repeat(3, 1, 1) # (N*3)x3x3
    # get first order SH
    feat_out_0 = features[..., :1] # Nx3x1

    # get second order SH
    # simply multiply using rotation matrix M
    src = features[..., 1:4].contiguous().view(-1,3) # (N*3)x3
    #feat_out_1 = feat_out_1.view(-1, 3, 1) # (N*3)x3x1
    #feat_out_1 = M @ feat_out_1 # (N*3)x3x1
    #feat_out_1 = feat_out_1.view(-1, 3, 3) # Nx3x3
    feat_out_1 = torch.stack( 
        [ M[:, 1, 1] * src[:, 0] - M[:, 1, 2] * src[:, 1] + M[:, 1, 0] * src[:, 2],
	     -M[:, 2, 1] * src[:, 0] + M[:, 2, 2] * src[:, 1] - M[:, 2, 0] * src[:, 2],
	      M[:, 0, 1] * src[:, 0] - M[:, 0, 2] * src[:, 1] + M[:, 0, 0] * src[:, 2]
        ], dim=-1) # (N*3, 3)
    feat_out_1 = feat_out_1.view(-1, 3, 3)

    if features.shape[-1] == 4:
        return torch.cat([feat_out_0, feat_out_1], dim=-1).permute(0, 2, 1) #Nx3x4    

    # get third order SH
    # https://github.com/broepstorff/ShRotation/blob/master/ShRotation/ShRotateFast.cpp
    # constants
    s_c3 = 0.94617469575 # (3*sqrt(5))/(4*sqrt(pi))
    s_c4 = -0.31539156525 # (-sqrt(5))/(4*sqrt(pi))
    s_c5 = 0.54627421529 # (sqrt(15))/(4*sqrt(pi))
    s_c_scale = 1.0/0.91529123286551084 
    s_c_scale_inv = 0.91529123286551084
    s_rc2 = 1.5853309190550713*s_c_scale
    s_c4_div_c3 = s_c4/s_c3
    s_c4_div_c3_x2 = (s_c4/s_c3)*2.0
    s_scale_dst2 = s_c3 * s_c_scale_inv
    s_scale_dst4 = s_c5 * s_c_scale_inv
    # read third order
    x = features[..., 4:9].contiguous().view(-1) # (N*3)x5
    # sparse matrix multiply
    # (N*3,) 
    sh0 =  x[..., 3] + x[..., 4] + x[..., 4] - x[..., 1]
    sh1 =  x[..., 0] + s_rc2*x[..., 2] +  x[..., 3] + x[..., 4]
    sh2 =  x[..., 0]
    sh3 = -x[..., 3]
    sh4 = -x[..., 1]
    
    # Rotations.  R0 and R1 just use the raw matrix columns
	# (N*3,)
    r2x = M[:, 0, 0] + M[:, 0, 1]
    r2y = M[:, 1, 0] + M[:, 1, 1]
    r2z = M[:, 2, 0] + M[:, 2, 1]
    
    r3x = M[:, 0, 0] + M[:, 0, 2]
    r3y = M[:, 1, 0] + M[:, 1, 2]
    r3z = M[:, 2, 0] + M[:, 2, 2]
    
    r4x = M[:, 0, 1] + M[:, 0, 2]
    r4y = M[:, 1, 1] + M[:, 1, 2]
    r4z = M[:, 2, 1] + M[:, 2, 2]
    
    # dense matrix multiplication one column at a time
    # column 0
    sh0_x = sh0 * M[:, 0, 0]
    sh0_y = sh0 * M[:, 1, 0]
    d0 = sh0_x * M[:, 1, 0]
    d1 = sh0_y * M[:, 2, 0]
    d2 = sh0 * (M[:, 2, 0]**2 + s_c4_div_c3)
    d3 = sh0_x * M[:, 2, 0]
    d4 = sh0_x * M[:, 0, 0] - sh0_y * M[:, 1, 0]
    
    # column 1
    sh1_x = sh1 * M[:, 0, 2]
    sh1_y = sh1 * M[:, 1, 2]
    d0 += sh1_x * M[:, 1, 2]
    d1 += sh1_y * M[:, 2, 2]
    d2 += sh1 * (M[:, 2, 2]**2 + s_c4_div_c3)
    d3 += sh1_x * M[:, 2, 2]
    d4 += sh1_x * M[:, 0, 2] - sh1_y * M[:, 1, 2]
    
    # column 2
    sh2_x = sh2 * r2x
    sh2_y = sh2 * r2y
    d0 += sh2_x * r2y
    d1 += sh2_y * r2z
    d2 += sh2 * (r2z * r2z + s_c4_div_c3_x2)
    d3 += sh2_x * r2z
    d4 += sh2_x * r2x - sh2_y * r2y

    # column 3
    sh3_x = sh3 * r3x
    sh3_y = sh3 * r3y
    d0 += sh3_x * r3y
    d1 += sh3_y * r3z
    d2 += sh3 * (r3z * r3z + s_c4_div_c3_x2)
    d3 += sh3_x * r3z
    d4 += sh3_x * r3x - sh3_y * r3y
    
    # column 4
    sh4_x = sh4 * r4x
    sh4_y = sh4 * r4y
    d0 += sh4_x * r4y
    d1 += sh4_y * r4z
    d2 += sh4 * (r4z * r4z + s_c4_div_c3_x2)
    d3 += sh4_x * r4z
    d4 += sh4_x * r4x - sh4_y * r4y

	# extra multipliers
    feat_out_2 = torch.stack([
        d0, 
        -d1, 
        d2 * s_scale_dst2,
        -d3,
        d4 * s_scale_dst4
    ], dim=-1) # (N*3)x5
    feat_out_2 = feat_out_2.view(-1, 3, 5)
    
    if features_left is None:
        return torch.cat([feat_out_0, feat_out_1, feat_out_2], dim=-1).permute(0, 2, 1) #Nx3x9    
    else:
        return torch.cat([feat_out_0, feat_out_1, feat_out_2, features_left], dim=-1).permute(0, 2, 1)

# L: an int
# xyz: a tensor [Bx3]
def get_basis_SH(L: int, xyz: torch.Tensor):
    x = xyz[:, :1]
    y = xyz[:, 1:2]
    z = xyz[:, 2:]
    devpi = 1./ math.sqrt(torch.pi)
    if L == 0:
        return torch.ones((xyz.shape[0], 1), device=xyz.device) * (0.5 * devpi)
    elif L == 1:
        return torch.cat([
            -math.sqrt(3) * y * devpi /2.,
            math.sqrt(3) * z * devpi / 2.,
            -math.sqrt(3) * x * devpi /2.,
        ], dim=-1)
    elif L == 2:
        return torch.cat([
            math.sqrt(15) * y * x * devpi / 2.,
            -math.sqrt(15) * y * z * devpi / 2.,
            math.sqrt(5) * (3*z**2 - 1.) * devpi / 4.,
            -math.sqrt(15) * x * z * devpi / 2.,
            math.sqrt(15) * (x**2 - y**2) * devpi / 4.,
        ], dim=-1)
    elif L == 3:
        return torch.cat([
            -math.sqrt(70) * y * (3*x**2 - y**2) * devpi / 8.,
            math.sqrt(105) * y * x * z * devpi / 2.,
            -math.sqrt(42) * y * (-1. + 5. * z**2) * devpi / 8.,
            math.sqrt(7) * z * (5 * z**2 - 3.) * devpi / 4.,
            -math.sqrt(42) * x * (-1. + 5. * z**2) * devpi / 8.,
            math.sqrt(105) * (x**2 - y**2) * z * devpi / 4.,
            -math.sqrt(70) * x * (x**2 - 3 * y**2) * devpi / 8.
        ], dim=-1)
    else:
        assert False, "Not implemented for now"

# q: Nx4
# features: Nxn_featx3, 3 here means 3 channel for rgb color
def batch_SH_rotate(q, features):
    #assert False, features.shape
    features = features.permute(0, 2, 1)
    if features.shape[-1] > 16:
        features_left = features[..., 16:]
    else:
        #1
        #1+3=4
        #1+3+5=9
        #1+3+5+7=16
        features_left = None
        assert features.shape[-1] in [1, 4, 9, 16], "At max support SH order of 3"
    
    if features.shape[-1] == 1:
        return features.permute(0, 2, 1) # order 1 is rotation-invariant; don't need to change anything 

    # get rotation matrix M:Nx3x3 given quaternion x
    # https://www.songho.ca/opengl/gl_quaternion.html
    # Nx1
    M = build_rotation(q) # Nx3x3
    M = M.repeat(3, 1, 1) # (N*3)x3x3
    # get first order SH
    feat_out_0 = features[..., :1] # Nx3x1

    # get second order SH
    # simply multiply using rotation matrix M
    src = features[..., 1:4].contiguous().view(-1,3) # (N*3)x3
    #feat_out_1 = feat_out_1.view(-1, 3, 1) # (N*3)x3x1
    #feat_out_1 = M @ feat_out_1 # (N*3)x3x1
    #feat_out_1 = feat_out_1.view(-1, 3, 3) # Nx3x3
    #feat_out_1 = torch.stack( 
    #    [ M[:, 1, 1] * src[:, 0] - M[:, 1, 2] * src[:, 1] + M[:, 1, 0] * src[:, 2],
	#     -M[:, 2, 1] * src[:, 0] + M[:, 2, 2] * src[:, 1] - M[:, 2, 0] * src[:, 2],
	#      M[:, 0, 1] * src[:, 0] - M[:, 0, 2] * src[:, 1] + M[:, 0, 0] * src[:, 2]
    #    ], dim=-1) # (N*3, 3)
    #print("replace here with a quaternion mulitp")
    
    #M: (Nx3)x3x3
    feat_out_1 = torch.stack( 
        [ M[:, 1, 1] * src[:, 0] - M[:, 1, 2] * src[:, 1] + M[:, 1, 0] * src[:, 2],
	     -M[:, 2, 1] * src[:, 0] + M[:, 2, 2] * src[:, 1] - M[:, 2, 0] * src[:, 2],
	      M[:, 0, 1] * src[:, 0] - M[:, 0, 2] * src[:, 1] + M[:, 0, 0] * src[:, 2]
        ], dim=-1) # (N*3, 3)
    feat_out_1 = feat_out_1.view(-1, 3, 3)
    
    #feat_out_1 = M @ src #(Nx3)x3x1
    #feat_out_1 = feat_out_1.view(-1, 3, 3)

    #if features.shape[-1] == 4:
    #    return torch.cat([feat_out_0, feat_out_1], dim=-1).permute(0, 2, 1) #Nx3x4    
    #else:
    #    return torch.cat([feat_out_0, feat_out_1, features[..., 4:]], dim=-1).permute(0, 2, 1)
    
    k = 1./math.sqrt(2)
    norms = torch.Tensor([
        [1., 0., 0.],
        [0., 0., 1.],
        [k, k, 0.],
        [k, 0., k],
        [0., k, k]
    ]).cuda() #5x3
    
    A = get_basis_SH(2, norms) # 5x5
    invA = torch.inverse(A)[None].expand(M.shape[0], 5, 5) # (Nx3)x5x5

    feat_out_2 = features[..., 4:9].contiguous().view(-1, 5, 1) # (Nx3)x5x1

    # M: (Nx3)x3x3
    new_norms = []
    for subnorm in norms: 
        subnorm = subnorm.view(1, 3, 1).expand(M.shape[0], 3, 1) # (Nx3)x3x1
        new_norms.append((M @ subnorm).view(-1, 1, 3)) # (Nx3)x1x3
    new_norms = torch.cat(new_norms, dim=1).view(-1, 3) # (Nx3x5)x3
    new_norms = get_basis_SH(2, new_norms).view(-1, 5, 5) # (Nx3)x5x5


    feat_out_2 = (new_norms @ invA @ feat_out_2).view(-1, 3, 5) # Nx3x5
    
    if features.shape[-1] == 9:
        return torch.cat([feat_out_0, feat_out_1, feat_out_2], dim=-1).permute(0, 2, 1) #Nx3x9    
    return  torch.cat([feat_out_0, feat_out_1, feat_out_2, features[..., 9:]], dim=-1).permute(0, 2, 1) #Nx3x9    
    k_ = 1./math.sqrt(3)
    norms_ = torch.Tensor([
        [k_, k_, k_],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
        [k, k, 0.],
        [k, 0., k],
        [0., k, k]
    ]).cuda() #7x3

    A_ = get_basis_SH(3, norms_) # 7x7
    invA_ = torch.inverse(A_)[None].expand(M.shape[0], 7, 7) # (Nx3)x7x7

    feat_out_3 = features[..., 9:16].contiguous().view(-1, 7, 1) # (Nx3)x7x1

    # M: (Nx3)x3x3
    new_norms_ = []
    for subnorm_ in norms_: 
        subnorm_ = subnorm_.view(1, 3, 1).expand(M.shape[0], 3, 1) # (Nx3)x3x1
        new_norms_.append((M @ subnorm_).view(-1, 1, 3)) # (Nx3)x1x3
    new_norms_ = torch.cat(new_norms_, dim=1).view(-1, 3) # (Nx3x7)x3
    new_norms_ = get_basis_SH(3, new_norms_).view(-1, 7, 7) # (Nx3)x7x7


    feat_out_3 = (new_norms_ @ invA_ @ feat_out_3).view(-1, 3, 7) # Nx3x7




    if features_left is None:
        return torch.cat([feat_out_0, feat_out_1, feat_out_2, feat_out_3], dim=-1).permute(0, 2, 1) #Nx3x9    
    else:
        return torch.cat([feat_out_0, feat_out_1, feat_out_2, feat_out_3, features_left], dim=-1).permute(0, 2, 1)

    #assert False, [norms.shape, A.shape]
    #assert False, torch.inverse(A).T

    # get third order SH
    # https://github.com/broepstorff/ShRotation/blob/master/ShRotation/ShRotateFast.cpp
    # constants
    s_c3 = 0.94617469575 # (3*sqrt(5))/(4*sqrt(pi))
    s_c4 = -0.31539156525 # (-sqrt(5))/(4*sqrt(pi))
    s_c5 = 0.54627421529 # (sqrt(15))/(4*sqrt(pi))
    s_c_scale = 1.0/0.91529123286551084 
    s_c_scale_inv = 0.91529123286551084
    s_rc2 = 1.5853309190550713*s_c_scale
    s_c4_div_c3 = s_c4/s_c3
    s_c4_div_c3_x2 = (s_c4/s_c3)*2.0
    s_scale_dst2 = s_c3 * s_c_scale_inv
    s_scale_dst4 = s_c5 * s_c_scale_inv
    # read third order
    x = features[..., 4:9].contiguous().view(-1) # (N*3)x5
    # sparse matrix multiply
    # (N*3,) 
    sh0 =  x[..., 3] + x[..., 4] + x[..., 4] - x[..., 1]
    sh1 =  x[..., 0] + s_rc2*x[..., 2] +  x[..., 3] + x[..., 4]
    sh2 =  x[..., 0]
    sh3 = -x[..., 3]
    sh4 = -x[..., 1]
    
    # Rotations.  R0 and R1 just use the raw matrix columns
	# (N*3,)
    r2x = M[:, 0, 0] + M[:, 0, 1]
    r2y = M[:, 1, 0] + M[:, 1, 1]
    r2z = M[:, 2, 0] + M[:, 2, 1]
    
    r3x = M[:, 0, 0] + M[:, 0, 2]
    r3y = M[:, 1, 0] + M[:, 1, 2]
    r3z = M[:, 2, 0] + M[:, 2, 2]
    
    r4x = M[:, 0, 1] + M[:, 0, 2]
    r4y = M[:, 1, 1] + M[:, 1, 2]
    r4z = M[:, 2, 1] + M[:, 2, 2]
    
    # dense matrix multiplication one column at a time
    # column 0
    sh0_x = sh0 * M[:, 0, 0]
    sh0_y = sh0 * M[:, 1, 0]
    d0 = sh0_x * M[:, 1, 0]
    d1 = sh0_y * M[:, 2, 0]
    d2 = sh0 * (M[:, 2, 0]**2 + s_c4_div_c3)
    d3 = sh0_x * M[:, 2, 0]
    d4 = sh0_x * M[:, 0, 0] - sh0_y * M[:, 1, 0]
    
    # column 1
    sh1_x = sh1 * M[:, 0, 2]
    sh1_y = sh1 * M[:, 1, 2]
    d0 += sh1_x * M[:, 1, 2]
    d1 += sh1_y * M[:, 2, 2]
    d2 += sh1 * (M[:, 2, 2]**2 + s_c4_div_c3)
    d3 += sh1_x * M[:, 2, 2]
    d4 += sh1_x * M[:, 0, 2] - sh1_y * M[:, 1, 2]
    
    # column 2
    sh2_x = sh2 * r2x
    sh2_y = sh2 * r2y
    d0 += sh2_x * r2y
    d1 += sh2_y * r2z
    d2 += sh2 * (r2z * r2z + s_c4_div_c3_x2)
    d3 += sh2_x * r2z
    d4 += sh2_x * r2x - sh2_y * r2y

    # column 3
    sh3_x = sh3 * r3x
    sh3_y = sh3 * r3y
    d0 += sh3_x * r3y
    d1 += sh3_y * r3z
    d2 += sh3 * (r3z * r3z + s_c4_div_c3_x2)
    d3 += sh3_x * r3z
    d4 += sh3_x * r3x - sh3_y * r3y
    
    # column 4
    sh4_x = sh4 * r4x
    sh4_y = sh4 * r4y
    d0 += sh4_x * r4y
    d1 += sh4_y * r4z
    d2 += sh4 * (r4z * r4z + s_c4_div_c3_x2)
    d3 += sh4_x * r4z
    d4 += sh4_x * r4x - sh4_y * r4y

	# extra multipliers
    feat_out_2 = torch.stack([
        d0, 
        -d1, 
        d2 * s_scale_dst2,
        -d3,
        d4 * s_scale_dst4
    ], dim=-1) # (N*3)x5
    feat_out_2 = feat_out_2.view(-1, 3, 5)
    
    if features_left is None:
        return torch.cat([feat_out_0, feat_out_1, feat_out_2], dim=-1).permute(0, 2, 1) #Nx3x9    
    else:
        return torch.cat([feat_out_0, feat_out_1, feat_out_2, features_left], dim=-1).permute(0, 2, 1)




# adopted from 4DGaussians
def batch_quaternion_multiply(q1, q2):
    """
    Multiply batches of quaternions.
    
    Args:
    - q1 (torch.Tensor): A tensor of shape [N, 4] representing the first batch of quaternions.
    - q2 (torch.Tensor): A tensor of shape [N, 4] representing the second batch of quaternions.
    
    Returns:
    - torch.Tensor: The resulting batch of quaternions after applying the rotation.
    """
    # Calculate the product of each quaternion in the batch
    w = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    x = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    y = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    z = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]

    # Combine into new quaternions
    q3 = torch.stack((w, x, y, z), dim=1)
    
    # Normalize the quaternions
    norm_q3 = q3 / torch.norm(q3, dim=1, keepdim=True)
    
    return norm_q3


# These are for SE mode
def skew(w):
    #return w[..., None].repeat(1, 1, 3)
    result = torch.zeros((w.shape[0], 3, 3), device=w.device)
    result[:, 0, 1] = -w[:, 2]
    result[:, 0, 2] = w[:, 1]
    result[:, 1, 0] = w[:, 2]
    result[:, 1, 2] = -w[:, 0]
    result[:, 2, 0] = -w[:, 1]
    result[:, 2, 1] = w[:, 0]
    return result

def exp_so3(W, theta): #W: Nx3x3, theta: N
    #W = skew(w)
    cp1 = torch.eye(3, device=W.device)[None, ...].repeat(W.shape[0], 1, 1) 
    #assert False, [theta.shape, W.shape]
    cp2 = torch.sin(theta).view(-1, 1, 1) * W 
    cp3 = (1.0 - torch.cos(theta)).view(-1, 1, 1) * W @ W 
    #assert False, [cp1.shape, cp2.shape, cp3.shape]
    return cp1 + cp2 + cp3

def rp_to_se3(R, p):
    #p = p.view(-1, 3, 1)
    #assert False, [torch.cat([R, p], dim=-1).shape, torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(p.device).repeat(R.shape[0], 1, 1).shape]
    return torch.cat([torch.cat([R, p], dim=-1), torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(p.device).repeat(R.shape[0], 1, 1)], dim=1)

def exp_se3(S, theta):
    w, v = S[:, :3], S[:, 3:]
    W = skew(w) #[r]_x
    #return rp_to_se3(w.view(-1, 3, 1).repeat(1, 1, 3), v.view(-1, 3, 1))
    R = exp_so3(W, theta)
    p = (theta.view(-1, 1, 1) * torch.eye(3, device=w.device)[None, ...].repeat(w.shape[0], 1, 1) + (1.0 - torch.cos(theta)).view(-1, 1, 1) * W +
       (theta - torch.sin(theta)).view(-1, 1, 1) * W @ W) @ v.view(-1, 3, 1)
    return rp_to_se3(R, p)

def to_homogenous(v):#Nx3
  return torch.cat([v, torch.ones_like(v[..., :1], device=v.device)], dim=-1).view(-1 ,4, 1)


def from_homogenous(v):
  return v[..., :3, 0] / v[..., -1:, 0]

class TemporalGaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, enable_offsh: bool, separate_offopa: bool, separate_offsh: bool,
        enable_static: bool, init_mode_gaussian: bool, stop_gradient: bool, use_skips: bool,
        new_deform: bool, shrink_lr: bool, use_nte: bool, use_SE: bool,
        anneal_band: bool, anneal_band_time: bool, anneal_band_steps: float,
        mult_quaternion: bool, rotate_sh: bool,
        posbase_pe, timebase_pe,
        defor_depth, net_width, 
        dynamic_sep: bool, use_ResFields: bool, ResField_mode: str,
        capacity: int, ewa_prune: bool):
        super().__init__()
        self.ewa_prune = ewa_prune
        self.use_ResFields = use_ResFields
        self.capacity = capacity
        self.ResField_mode = ResField_mode
        self.dynamic_sep = dynamic_sep
        self.use_SE = use_SE
        self.mult_quaternion = mult_quaternion
        self.rotate_sh = rotate_sh
        self.anneal_band = anneal_band
        if self.anneal_band:
            self.warp_alpha_sched = LinearSchedule(
                initial_value = 0.0,
                final_value = posbase_pe, 
                num_steps=anneal_band_steps
            )
        self.anneal_band_time = anneal_band_time
        if self.anneal_band_time:
            self.time_alpha_sched = LinearSchedule(
                initial_value = 0.0,
                final_value = timebase_pe, 
                num_steps=anneal_band_steps
            )
        self.new_deform = new_deform
        self.shrink_lr = shrink_lr
        self.stop_gradient = stop_gradient
        self.enable_static = enable_static
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._features_dino = torch.empty(0)
        # self._features_clip = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._isstatic = torch.empty(0) # to store whether need deform or not 
        self.max_radii2D = torch.empty(0)
        self.max_scaling = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.opacity_accum = torch.empty(0)
        self.scaling_accum = torch.empty(0)
        self.xyz_motion_accum = torch.empty(0)

        self.denom = torch.empty(0)
        self.optimizer = None
        #self.defor_optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.posbase_pe = posbase_pe
        self.timebase_pe = timebase_pe
        times_ch = 2*timebase_pe+1
        pts_ch = 3+3*posbase_pe*2
        self.defor_depth = defor_depth
        self.net_width = net_width
        self.enable_offsh = enable_offsh
        self.deformation_net = Deformation(
            W=net_width, D=defor_depth, 
            input_ch=pts_ch, 
            input_ch_time=times_ch,
            max_sh_degree=sh_degree if enable_offsh else None,
            init_mode_gaussian=init_mode_gaussian,
            skips=[defor_depth//2] if use_skips else [],
            use_nte=use_nte, use_SE=use_SE,
            use_ResFields=use_ResFields,
            mode=self.ResField_mode,
            capacity=self.capacity).to("cuda")
        
        self.separate_offopa = separate_offopa
        if self.separate_offopa:
            self.opa_net = Deformation(
                W=net_width, D=defor_depth, 
                input_ch=pts_ch, 
                input_ch_time=times_ch,
                opa_only=True,
                init_mode_gaussian=init_mode_gaussian,
                skips=[defor_depth//2] if use_skips else [],
                use_nte=use_nte, use_SE=use_SE,use_ResFields=use_ResFields,
                mode=self.ResField_mode,
                capacity=self.capacity
            ).to('cuda')
        self.separate_offsh = separate_offsh
        if self.separate_offsh:
            self.sh_net = Deformation(
                W=net_width, D=defor_depth,
                input_ch=pts_ch,
                input_ch_time=times_ch,
                max_sh_degree=sh_degree,
                sh_only=True,
                init_mode_gaussian=init_mode_gaussian,
                skips=[defor_depth//2] if use_skips else [],
                use_nte=use_nte, use_SE=use_SE,use_ResFields=use_ResFields,
                mode=self.ResField_mode,
                capacity=self.capacity
            ).to("cuda")    
        #self.time_poc = torch.empty(0)
        #self.pos_poc = torch.empty(0)
        self.register_buffer('time_poc', torch.cuda.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.cuda.FloatTensor([(2**i) for i in range(posbase_pe)]))
        


        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._features_dino,
            # self._features_clip
            self._scaling,
            self._rotation,
            self._opacity,
            self._isstatic,
            self.max_radii2D,
            self.max_scaling,
            self.xyz_gradient_accum,
            self.opacity_accum,
            self.scaling_accum,
            self.xyz_motion_accum,
            self.denom,
            self.optimizer.state_dict(),
            #self.defor_optimizer.state_dict(),
            self.spatial_lr_scale,
            self.posbase_pe,
            self.timebase_pe,
            self.defor_depth,
            self.net_width,
            self.deformation_net.state_dict(),
            self.opa_net.state_dict() if self.separate_offopa else None,
            self.sh_net.state_dict() if self.separate_offsh else None
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._features_dino,
        # self._features_clip,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._isstatic,
        self.max_radii2D, 
        self.max_scaling,
        xyz_gradient_accum, 
        opacity_accum,
        scaling_accum,
        xyz_motion_accum,
        denom,
        opt_dict, 
        #opt_defor_dict,
        self.spatial_lr_scale,
        posbase_pe,
        timebase_pe,
        defor_depth,
        net_width,
        defor_dict,
        opa_dict,
        sh_dict
        ) = model_args

        if posbase_pe != self.posbase_pe or timebase_pe != self.timebase_pe or defor_depth != self.defor_depth or net_width != self.net_width:
            assert False, "restoring a model that does not match deformation network structure!"
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.opacity_accume = opacity_accum
        self.scaling_accum = scaling_accum
        self.xyz_motion_accum = xyz_motion_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        #self.defor_optimizer.load_state_dict(opt_defor_dict)

        #times_ch = 2*self.timebase_pe+1
        #pts_ch = 3+3*self.posbase_pe*2
        #self.deformation_net = Deformation(
        #    W=self.net_width, D=self.defor_depth, 
        #    input_ch=pts_ch, 
        #    input_ch_time=times_ch)
        self.deformation_net.load_state_dict(defor_dict)
        #self.time_poc = torch.FloatTensor([(2**i) for i in range(self.timebase_pe)]).to("cuda")
        #self.pos_poc = torch.FloatTensor([(2**i) for i in range(self.posbase_pe)]).to("cuda")
        if opa_dict is not None:
            self.opa_net.load_state_dict(opa_dict)       
        if sh_dict is not None:
            self.sh_net.load_state_dict(sh_dict) 

    # times_sel: a float with value [0, 1]
    def pass_deform(self, times_sel, window=None, window_time=None):
        if self.use_ResFields:
            times_sel, frame_id = times_sel
        #if self.use_ResFields:
        #    times, frame_id = times_sel
        #    times_emb = (torch.FloatTensor([times]).view(1,).repeat(self._xyz.shape[0]).to(self._xyz.device),
        #        torch.FloatTensor([frame_id]).long().view(1,).repeat(self._xyz.shape[0]).to(self._xyz.device) if frame_id is not None else None)
        #    pts_emb = self._xyz.detach() if self.stop_gradient else self._xyz
        #    return self.deformation_net(pts_emb, times_emb)
        #assert len(times_self.shape) == 1 and times_self.shape[0] == 1
        times_sel = torch.FloatTensor([times_sel]).view(1, 1).repeat(self._xyz.shape[0], 1).to(self._xyz.device)
        #assert False, [times_sel.device, self.time_poc.device, self._xyz.device, self.pos_poc.device]
        times_emb = poc_fre(times_sel, self.time_poc)
        if window_time is not None:
            window_time = window_time.view(1, window_time.shape[0], 1, 1).repeat(times_emb.shape[0], 1, 2, 1).view(times_emb.shape[0], -1)
            times_emb[:, 1:] *=  window_time
        pts_emb = poc_fre(self._xyz, self.pos_poc)
        if window is not None:
            #identity, pts_emb_ = torch.split(pts_emb, (3,pts_emb.shape[-1]-3), dim=-1)
            #print(pts_emb.shape)
            #pts_emb_ = pts_emb_.view(pts_emb.shape[0], -1, 2, 3)
            #print(pts_emb.shape)
            #assert False, "Pause here; missing batch?"
            #pts_emb_ = pts_emb_ * window.view(1, pts_emb.shape[1], 1, 1)
            #pts_emb = torch.cat([identity, pts_emb_.view(pts_emb.shape[0], -1)], dim=-1)
            window = window.view(1, window.shape[0], 1, 1).repeat(pts_emb.shape[0], 1, 2, 3).view(pts_emb.shape[0], -1)
            pts_emb[:, 3:] *=  window

        
        if self.stop_gradient:
            pts_emb = pts_emb.detach()
        if self.use_ResFields:
            times_emb = (times_sel, times_emb, frame_id)
        return  self.deformation_net(pts_emb, times_emb)
        #return dpos, dscale, drot, dopaq
    # times_sel: a float with value [0, 1]
    def pass_opa(self, times_sel, window=None, window_time=None):
        if self.use_ResFields:
            times, frame_id = times_sel
        #    times_emb = (torch.FloatTensor([times]).view(1,).repeat(self._xyz.shape[0]).to(self._xyz.device),
        #        torch.FloatTensor([frame_id]).long().view(1,).repeat(self._xyz.shape[0]).to(self._xyz.device) if frame_id is not None else None)
        #    pts_emb = self._xyz.detach() if self.stop_gradient else self._xyz
        #    return self.opa_net(pts_emb, times_emb)
        #assert len(times_self.shape) == 1 and times_self.shape[0] == 1
        times_sel = torch.FloatTensor([times_sel]).view(1, 1).repeat(self._xyz.shape[0], 1).to(self._xyz.device)
        #assert False, [times_sel.device, self.time_poc.device, self._xyz.device, self.pos_poc.device]
        times_emb = poc_fre(times_sel, self.time_poc)
        if window_time is not None:
            window_time = window_time.view(1, window_time.shape[0], 1, 1).repeat(times_emb.shape[0], 1, 2, 1).view(times_emb.shape[0], -1)
            times_emb[:, 1:] *=  window_time
        pts_emb = poc_fre(self._xyz, self.pos_poc)
        
        if window is not None:
            #identity, pts_emb_ = torch.split(pts_emb, (3,pts_emb.shape[-1]-3), dim=-1)
            #print(pts_emb.shape)
            #pts_emb_ = pts_emb_.view(pts_emb.shape[0], -1, 2, 3)
            #print(pts_emb.shape)
            #assert False, "Pause here; missing batch?"
            #pts_emb_ = pts_emb_ * window.view(1, pts_emb.shape[1], 1, 1)
            #pts_emb = torch.cat([identity, pts_emb_.view(pts_emb.shape[0], -1)], dim=-1)
            window = window.view(1, window.shape[0], 1, 1).repeat(pts_emb.shape[0], 1, 2, 3).view(pts_emb.shape[0], -1)
            pts_emb[:, 3:] *=  window
        if self.stop_gradient:
            pts_emb = pts_emb.detach()
        if self.use_ResFields:
            times_emb = (times_sel, times_emb, frame_id)
        return  self.opa_net(pts_emb, times_emb)
        #return dpos, dscale, drot, dopaq

    def pass_sh(self, times_sel, window=None, window_time=None):
        if self.use_ResFields:
            times, frame_id = times_sel
        #    times_emb = (torch.FloatTensor([times]).view(1,).repeat(self._xyz.shape[0]).to(self._xyz.device),
        #        torch.FloatTensor([frame_id]).long().view(1,).repeat(self._xyz.shape[0]).to(self._xyz.device) if frame_id is not None else None)
        #    pts_emb = self._xyz.detach() if self.stop_gradient else self._xyz
        #    return self.sh_net(pts_emb, times_emb)
        #assert len(times_self.shape) == 1 and times_self.shape[0] == 1
        times_sel = torch.FloatTensor([times_sel]).view(1, 1).repeat(self._xyz.shape[0], 1).to(self._xyz.device)
        #assert False, [times_sel.device, self.time_poc.device, self._xyz.device, self.pos_poc.device]
        times_emb = poc_fre(times_sel, self.time_poc)
        if window_time is not None:
            window_time = window_time.view(1, window_time.shape[0], 1, 1).repeat(times_emb.shape[0], 1, 2, 1).view(times_emb.shape[0], -1)
            times_emb[:, 1:] *=  window_time
        pts_emb = poc_fre(self._xyz, self.pos_poc)
        if window is not None:
            #identity, pts_emb_ = torch.split(pts_emb, (3,pts_emb.shape[-1]-3), dim=-1)
            #print(pts_emb.shape)
            #pts_emb_ = pts_emb_.view(pts_emb.shape[0], -1, 2, 3)
            #print(pts_emb.shape)
            #assert False, "Pause here; missing batch?"
            #pts_emb_ = pts_emb_ * window.view(1, pts_emb.shape[1], 1, 1)
            #pts_emb = torch.cat([identity, pts_emb_.view(pts_emb.shape[0], -1)], dim=-1)
            window = window.view(1, window.shape[0], 1, 1).repeat(pts_emb.shape[0], 1, 2, 3).view(pts_emb.shape[0], -1)
            pts_emb[:, 3:] *=  window
        
        if self.stop_gradient:
            pts_emb = pts_emb.detach()
        if self.use_ResFields:
            times_emb = (times_sel, times_emb, frame_id)
        return  self.sh_net(pts_emb, times_emb)
        #return dpos, dscale, drot, dopaq
    
    # TODO: feature deformation
    def get_deformed_no_opaq(self, times_sel, disable_offscale, disable_offopa, disable_morph, multiply_offopa, window=None, window_time=None):
        assert not self.separate_offopa, "OpaNet exists!"
        if self.use_SE:
            if self.separate_offsh:
                dfeat_dc, dfeat_extra = self.pass_sh(times_sel, window=window, window_time=window_time)
                w, v, dscale, drot, dopaq = self.pass_deform(times_sel, window=window, window_time=window_time)
                #shs = features_dc = self._features_dc
                #features_rest = self._features_rest
                #return torch.cat((self._features_dc, self._features_rest), dim=1)
                #shs = torch.cat((
                #    self._features_dc + dfeat_dc,
                #    self._features_rest + dfeat_extra
                #))
            elif self.enable_offsh:
                w, v, dscale, drot, dopaq, dfeat = self.pass_deform(times_sel, window=window, window_time=window_time)
            else:
                w, v, dscale, drot, dopaq = self.pass_deform(times_sel, window=window, window_time=window_time)
            #if self.separate_offsh:
            theta = torch.norm(w, dim=-1).detach()
            w /= theta[:, None]
            v /= theta[:, None]
            screw_axis = torch.cat([w, v], dim=-1)
            transform = exp_se3(screw_axis, theta)
            #assert False, [transform.shape, to_homogenous(self._xyz_dy).shape, ]
            means3D = from_homogenous(
                    transform @ to_homogenous(self._xyz))
            dpos = means3D - self._xyz        
        else:
            if self.separate_offsh:
                dfeat_dc, dfeat_extra = self.pass_sh(times_sel, window=window, window_time=window_time)
                dpos, dscale, drot, dopaq = self.pass_deform(times_sel, window=window, window_time=window_time)
                #shs = features_dc = self._features_dc
                #features_rest = self._features_rest
                #return torch.cat((self._features_dc, self._features_rest), dim=1)
                #shs = torch.cat((
                #    self._features_dc + dfeat_dc,
                #    self._features_rest + dfeat_extra
                #))
            elif self.enable_offsh:
                dpos, dscale, drot, dopaq, dfeat = self.pass_deform(times_sel, window=window, window_time=window_time)
            else:
                dpos, dscale, drot, dopaq = self.pass_deform(times_sel, window=window, window_time=window_time)
            #if self.separate_offsh:
            
        means3D = self._xyz + dpos
        if disable_morph:
            scales = self.scaling_activation(self._scaling)
            rotations = self.rotation_activation(self._rotation)
            dscale *= 0
            drot *= 0
        else:
            if disable_offscale:
                dscale *= 0
            if self.new_deform:
                scales = self.scaling_activation(self._scaling) + dscale
                if self.mult_quaternion:
                    rotations = batch_quaternion_multiply(self._rotation, drot)
                else:
                    rotations = self.rotation_activation(self._rotation) + drot
                
            else:
                scales = self.scaling_activation(self._scaling + dscale)
                if self.mult_quaternion:
                    rotations = batch_quaternion_multiply(self._rotation, drot)
                else:
                    rotations = self.rotation_activation(self._rotation + drot)
        if disable_offopa:
            opacities = self.opacity_activation(self._opacity)
        elif multiply_offopa:
            opacities = self.opacity_activation(self._opacity * (1e-16+dopaq))
        else:
            if self.new_deform:
                opacities = self.opacity_activation(self._opacity) + dopaq
            else:
                opacities = self.opacity_activation(self._opacity + dopaq)
        if self.separate_offsh:
            assert False, "not supporting motion regularization"
            return means3D, opacities, scales, rotations, dfeat_dc, dfeat_extra
        elif self.enable_offsh:
            #assert False, "not supporting motion regularization"
            return means3D, opacities, scales, rotations, dfeat, dpos, dscale, drot   
        else:
            return means3D, opacities, scales, rotations, dpos, dscale, drot

    def get_deformed_opaq(self, times_sel, disable_offscale, disable_offopa, disable_morph, multiply_offopa, window=None, window_time=None):
        #assert False, "not supporting motion regularization"
        assert self.separate_offopa, "OpaNet does not exist!"
        if self.use_SE:
            if self.separate_offsh:
                dfeat_dc, dfeat_extra = self.pass_sh(times_sel, window=window, window_time=window_time)
                w, v, dscale, drot, _ = self.pass_deform(times_sel, window=window, window_time=window_time)
                #shs = features_dc = self._features_dc
                #features_rest = self._features_rest
                #return torch.cat((self._features_dc, self._features_rest), dim=1)
                #shs = torch.cat((
                #    self._features_dc + dfeat_dc,
                #    self._features_rest + dfeat_extra
                #))
            elif self.enable_offsh:
                w, v, dscale, drot, _, dfeat = self.pass_deform(times_sel, window=window, window_time=window_time)
            else:
                w, v, dscale, drot, _ = self.pass_deform(times_sel, window=window, window_time=window_time)
            #if self.separate_offsh:
            theta = torch.norm(w, dim=-1).detach()
            w /= theta[:, None]
            v /= theta[:, None]
            screw_axis = torch.cat([w, v], dim=-1)
            transform = exp_se3(screw_axis, theta)
            #assert False, [transform.shape, to_homogenous(self._xyz_dy).shape, ]
            means3D = from_homogenous(
                    transform @ to_homogenous(self._xyz))
            dpos = means3D - self._xyz        
        else:
            if self.separate_offsh:
                dfeat_dc, dfeat_extra = self.pass_sh(times_sel, window=window, window_time=window_time)
                dpos, dscale, drot, _ = self.pass_deform(times_sel, window=window, window_time=window_time)
            elif self.enable_offsh:
                dpos, dscale, drot, _, dfeat = self.pass_deform(times_sel, window=window, window_time=window_time)
            else:
                dpos, dscale, drot, _ = self.pass_deform(times_sel, window=window, window_time=window_time)
            #if self.separate_offsh:
        #    dfeat = self.pass_sh(times_sel)
        dopaq = self.pass_opa(times_sel, window=window, window_time=window_time)
        means3D = self._xyz + dpos
        if disable_morph:
            scales = self.scaling_activation(self._scaling)
            rotations = self.rotation_activation(self._rotation)
            dscale *= 0
            drot *= 0
        else:
            if disable_offscale:
                dscale *= 0
            if self.new_deform:
                scales = self.scaling_activation(self._scaling) + dscale
                if self.mult_quaternion:
                    rotations = batch_quaternion_multiply(self._rotation, drot)
                else:
                    rotations = self.rotation_activation(self._rotation) + drot
            else:
                scales = self.scaling_activation(self._scaling + dscale)
                if self.mult_quaternion:
                    rotations = batch_quaternion_multiply(self._rotation, drot)
                else:
                    rotations = self.rotation_activation(self._rotation + drot)
        #if disable_offopa:
        #    opacities = self.opacity_activation(self._opacity)
        #if multiply_offopa:
        #    opacities = self.opacity_activation(self._opacity * (1e-16+dopaq))
        #else:
        if self.new_deform:
            opacities = self.opacity_activation(self._opacity) + dopaq
        else:
            opacities = self.opacity_activation(self._opacity + dopaq)
        if self.separate_offsh:
            assert False, "not supporting motion regularization"
            return means3D, opacities, scales, rotations, dfeat_dc, dfeat_extra
        elif self.enable_offsh:
            #assert False, "not supporting motion regularization"
            return means3D, opacities, scales, rotations, dfeat, dpos, dscale, drot    
        else:
            return means3D, opacities, scales, rotations, dpos, dscale, drot 
    def get_deformed(self, times_sel, frame_id, disable_offscale, disable_offopa, disable_morph, multiply_offopa, anneal_band_iteration):
        if self.use_ResFields:
            times_sel = (times_sel, frame_id)
        if self.anneal_band:
            assert anneal_band_iteration is not None
            window = cosine_easing_window(min_freq_log2=None, max_freq_log2=None, num_bands=self.posbase_pe, 
                alpha=self.warp_alpha_sched.get(anneal_band_iteration))
            #assert False, window.shape
        else:
            window = None
        if self.anneal_band_time:
            assert anneal_band_iteration is not None
            window_time = cosine_easing_window(min_freq_log2=None, max_freq_log2=None, num_bands=self.timebase_pe, 
                alpha=self.time_alpha_sched.get(anneal_band_iteration))
        else:
            window_time = None
        if self.separate_offopa:
            return self.get_deformed_opaq(times_sel, disable_offscale, disable_offopa, disable_morph, multiply_offopa, window=window, window_time=window_time)
        else:
            return self.get_deformed_no_opaq(times_sel, disable_offscale, disable_offopa, disable_morph, multiply_offopa, window=window, window_time=window_time)
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dino(self):
        return self._features_dino
    
    # @property
    # def get_features_clip(self):
    #     return self._features_clip
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    #def get_covariance(self, scaling_modifier = 1):
    #    return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd_dy(self, pcd : BasicPointCloud, spatial_lr_scale : float, pcd_dy: BasicPointCloud):
        self.spatial_lr_scale = spatial_lr_scale
        
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
        fused_point_cloud_dy = torch.tensor(np.asarray(pcd_dy.points)).float().cuda()
        fused_color_dy = RGB2SH(torch.tensor(np.asarray(pcd_dy.colors)).float().cuda())

        features = torch.zeros((fused_color.shape[0]+fused_color_dy.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:fused_color.shape[0], :3, 0 ] = fused_color
        features[fused_color.shape[0]:, :3, 0] = fused_color_dy
        features[:, 3:, 1:] = 0.0

        dino_features = torch.zeros((fused_color.shape[0]+fused_color_dy.shape[0], 3, 1)).float().cuda()
        # clip_features = torch.zeros((fused_color.shape[0]+fused_color_dy.shape[0], 3, 1)).float().cuda()

        print("Number of static points at initialisation : ", fused_point_cloud.shape[0])
        print("Number of dynamic points at initialisation : ", fused_point_cloud_dy.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))


        dist2_dy = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd_dy.points)).float().cuda()), 0.0000001)
        scales_dy = torch.log(torch.sqrt(dist2_dy))[...,None].repeat(1, 3)
        rots_dy = torch.zeros((fused_point_cloud_dy.shape[0], 4), device="cuda")
        rots_dy[:, 0] = 1

        opacities_dy = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud_dy.shape[0], 1), dtype=torch.float, device="cuda"))


        self._xyz = nn.Parameter(torch.cat([fused_point_cloud, fused_point_cloud_dy], dim=0).requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_dino = nn.Parameter(dino_features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_clip = nn.Parameter(clip_features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(torch.cat([scales, scales_dy], dim=0).requires_grad_(True))
        self._rotation = nn.Parameter(torch.cat([rots, rots_dy], dim=0).requires_grad_(True))
        self._opacity = nn.Parameter(torch.cat([opacities, opacities_dy], dim=0).requires_grad_(True))
        
        _isstatic = torch.cat([
            torch.zeros((fused_point_cloud.shape[0], 1)),
            torch.ones((fused_point_cloud_dy.shape[0], 1))], dim=0).cuda() 
        #is_zero = _isstatic < 0.5
        #_isstatic[is_zero] = 0
        #_isstatic[~is_zero] = 1
        self._isstatic = nn.Parameter(_isstatic.float().requires_grad_(False))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_scaling = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, pcd_dy=None):
        if pcd_dy is not None:
            self.create_from_pcd_dy(pcd, spatial_lr_scale, pcd_dy)
            return
        self.spatial_lr_scale = spatial_lr_scale
        
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        dino_features = torch.zeros((fused_color.shape[0], 3, 1)).float().cuda()
        # clip_features = torch.zeros((fused_color.shape[0], 3, 1)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))


        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_dino = nn.Parameter(dino_features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_clip = nn.Parameter(clip_features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        _isstatic = torch.rand((fused_point_cloud.shape[0], 1)).cuda() < 0.5 
        #is_zero = _isstatic < 0.5
        #_isstatic[is_zero] = 0
        #_isstatic[~is_zero] = 1
        self._isstatic = nn.Parameter(_isstatic.float().requires_grad_(False))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_scaling = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opacity_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scaling_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_motion_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            # TODO: training args for dino and clip feature lr
            {'params': [self._features_dino], 'lr': training_args.feature_lr, "name": "f_dino"},
            # {'params': [self._features_clip], 'lr': training_args.feature_lr, "name": "f_clip"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.deformation_net.parameters(), 'lr': training_args.defor_lr, "name": "deformation", 
            'weight_decay': training_args.defor_weight_decay},
            ]
        if self.separate_offopa:
            #assert False, "Not supported"
            l += [{'params': self.opa_net.parameters(), 'lr': training_args.opa_lr, "name": "opa_deform"}]
        if self.separate_offsh:
            assert False, "Not supported"
            l += [{'params': self.sh_net.parameters(), 'lr': training_args.sh_lr, "name": "sh_deform"}]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        #self.defor_optimizer = torch.optim.Adam(
        #    [{'params': self.deformation_net.parameters(), 'lr': training_args.defor_lr, "name": "deformation", 
        #    'weight_decay': training_args.defor_weight_decay},
        #    ], 
        #eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.scaling_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr,
                                                            lr_final=0.002*training_args.scaling_lr,
                                                            max_steps=training_args.scaling_lr_max_steps)
        self.rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr,
                                                            lr_final=0.002*training_args.rotation_lr,
                                                            max_steps=training_args.rotation_lr_max_steps)

        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.defor_lr,
                                                            lr_final=0.001*training_args.defor_lr,
                                                            max_steps=training_args.defor_lr_max_steps,
                                                            )
        
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr        
            if self.shrink_lr and param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group["lr"] = lr
            if self.shrink_lr and param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group["lr"] = lr
            #for param_group in self.defor_optimizer.param_groups:
            if param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group["lr"] = lr
            
        return 


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._features_dino.shape[1]*self._features_dino.shape[2]):
            l.append('f_dino_{}'.format(i))
        # for i in range(self._features_clip.shape[1]*self._features_clip.shape[2]):
        #     l.append('f_clip_{}'.format(i))
        l.append('opacity')
        l.append('isstatic')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dino = self._features_dino.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_clip = self._features_clip.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        isstatic = self._isstatic.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, isstatic, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        torch.save({
            "defor_dict": self.deformation_net.state_dict(),
            "opa_dict": self.opa_net.state_dict() if self.separate_offopa else None,
            "sh_dict": self.sh_net.state_dict() if self.separate_offsh else None,
            "timebase_pe": self.timebase_pe,
            "posbase_pe": self.posbase_pe,
            "net_width": self.net_width, 
            "defor_depth": self.defor_depth, 
            }, 
            path[:-4] + ".pth"
        )

    def reset_opacity(self):
       
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        if self.separate_offopa:
            assert not self.use_ResFields 
            self.opa_net.apply(self.opa_net._init_weights)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        isstatic = np.asarray(plydata.elements[0]["isstatic"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        
        features_dino = np.zeros((xyz.shape[0], 3, 1))
        features_dino[:, 0, 0] = np.asarray(plydata.elements[0]["f_dino_0"])
        features_dino[:, 1, 0] = np.asarray(plydata.elements[0]["f_dino_1"])
        features_dino[:, 2, 0] = np.asarray(plydata.elements[0]["f_dino_2"])

        # features_clip = np.zeros((xyz.shape[0], 3, 1))
        # features_clip[:, 0, 0] = np.asarray(plydata.elements[0]["f_clip_0"])
        # features_clip[:, 1, 0] = np.asarray(plydata.elements[0]["f_clip_1"])
        # features_clip[:, 2, 0] = np.asarray(plydata.elements[0]["f_clip_2"])
        
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_dino = nn.Parameter(torch.tensor(features_dino, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_clip = nn.Parameter(torch.tensor(features_clip, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._isstatic = nn.Parameter(torch.tensor(isstatic, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        defor_checkpoint = torch.load(path[:-4] + ".pth")
        #{
        #    "defor_dict": self.deformation_net.state_dict(),
        #    "timebase_pe": self.timebase_pe,
        #    "posbase_pe": self.posbase_pe,
        #    "net_width": self.net_width, 
        #    "defor_depth": self.defor_depth, 
        #},
        assert self.timebase_pe == defor_checkpoint["timebase_pe"] and self.posbase_pe == defor_checkpoint["posbase_pe"]\
        and self.net_width == defor_checkpoint["net_width"] and self.defor_depth == defor_checkpoint["defor_depth"], "does not match deformation net arch!"
        #times_ch = 2*self.timebase_pe+1
        #pts_ch = 3+3*self.posbase_pe*2
        #self.deformation_net = Deformation(
        #    W=self.net_width, D=self.defor_depth, 
        #    input_ch=pts_ch, 
        #    input_ch_time=times_ch)
        self.deformation_net.load_state_dict(defor_checkpoint["defor_dict"])
        if self.separate_offopa:
            self.opa_net.load_state_dict(defor_checkpoint["opa_dict"])
        if self.separate_offsh:
            self.sh_net.load_state_dict(defor_checkpoint["sh_dict"])
        #self.time_poc = torch.FloatTensor([(2**i) for i in range(self.timebase_pe)]).to("cuda")
        #self.pos_poc = torch.FloatTensor([(2**i) for i in range(self.posbase_pe)]).to("cuda")
        
        

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) != 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._features_dino = optimizable_tensors["f_dino"]
        # self._features_clip = optimizable_tensors["f_clip"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self._isstatic = nn.Parameter(self._isstatic[valid_points_mask].requires_grad_(False))
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.opacity_accum = self.opacity_accum[valid_points_mask]
        self.scaling_accum = self.scaling_accum[valid_points_mask]
        self.xyz_motion_accum = self.xyz_motion_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.max_scaling = self.max_scaling[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}

        
        for group in self.optimizer.param_groups:
            if len(group["params"]) != 1:
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_isstatic, new_scaling, new_rotation,
                              new_features_dino, new_features_clip=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "f_dino": new_features_dino,
        # "f_clip": new_features_clip,
        "opacity": new_opacities,
        #"isstatic": new_isstatic,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._features_dino = optimizable_tensors["f_dino"]
        # self._features_clip = optimizable_tensors["f_clip"]
        self._opacity = optimizable_tensors["opacity"]
        self._isstatic = nn.Parameter(torch.cat([self._isstatic, new_isstatic], dim=0).requires_grad_(False))
        #assert False
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opacity_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scaling_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_motion_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_scaling = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_features_dino = self._features_dino[selected_pts_mask].repeat(N,1,1)
        # new_features_clip = self._features_clip[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_isstatic = self._isstatic[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_isstatic, new_scaling, new_rotation,
                                   new_features_dino)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_features_dino = self._features_dino[selected_pts_mask]
        # new_features_clip = self._features_clip[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_isstatic = self._isstatic[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_isstatic, new_scaling, new_rotation,
                                   new_features_dino)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, dynamic_sep=False, min_motion=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if self.dynamic_sep and dynamic_sep:
            #assert False, "Not used for now"
            motions = self.xyz_motion_accum / self.denom
            known = ~(motions.isnan())
            # for determining value of min_motion
            
            #assert False, [torch.all(motions[known & (self._isstatic==0)] ==0.), motions[known & (self._isstatic==1)]]
            #left = motions[known]
            #assert False, [
            #    torch.quantile(left, 0.9),
            #    torch.quantile(left, 0.8),
            #    torch.quantile(left, 0.5),
            #    torch.quantile(left, 0.3),
            #    torch.quantile(left, 0.2),
            #    torch.quantile(left, 0.1)
            #]            
            #assert False, [motions.shape, motions.isnan().shape, (self._isstatic[:, 0]==0).shape]
            motions[motions.isnan() | (self._isstatic==0)] = -1.
            values = motions[motions != -1.]
            min_motion = torch.quantile(values, min_motion)

            # if certain gaussian's motion is unknown, do not change its _isstatic
            # otherwise if motion is smaller than min_motion, change _isstatic value to 0.
            print(f"observed points#: {float(torch.sum(known).cpu())}/{motions.shape[0]}")
            print(f"Dynamic motion: {float(torch.mean(motions[known & (self._isstatic==1)]))}")
            print(f"Static motion: {float(torch.mean(motions[known & (self._isstatic==0)]))}")
            print(f"Motion threshold: {float(min_motion)}")
            print(f"Maximum motion: {float(torch.max(motions))}, 90%: {float(torch.quantile(motions, 0.9))}")
            print(f"Translate Points#: {float(torch.sum((motions >=0.) & (motions < min_motion)))}")
            print("Before: ", float(torch.sum(self._isstatic).cpu()))
            self._isstatic[(motions >=0) & (motions < min_motion)] = 0.
            print("After: ", float(torch.sum(self._isstatic).cpu()))

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        
        if self.ewa_prune:
            opacities = self.opacity_accum / self.denom
            isnan = opacities.isnan().view(-1)
            opacities[isnan] = min_opacity + 1. # make sure not included in prune mask
            prune_mask = (opacities < min_opacity).squeeze()
            #prune_mask = prune_mask | isnan
            scalings = self.scaling_accum / self.denom
            scalings[isnan] = 0. # make sure smaller than threshold
            if max_screen_size:
                big_points_vs = self.max_radii2D > max_screen_size
                big_points_ws = (self.get_scaling.max(dim=1).values > 0.1 * extent) | (scalings.view(-1) > 0.1 * extent)
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        else:
            prune_mask = (self.get_opacity < min_opacity).squeeze()
            if max_screen_size:
                big_points_vs = self.max_radii2D > max_screen_size
                big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            
            


        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, opacity_tensor=None, scaling_tensor=None):
        #if self.dynamic_sep:
        #    return self.add_densification_stats_motion(viewspace_point_tensor, update_filter, motion_tensor)
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        if self.ewa_prune:
            self.opacity_accum[update_filter] += opacity_tensor[update_filter].view(-1, 1)
            self.scaling_accum[update_filter] += scaling_tensor[update_filter].view(-1, 1)
        self.denom[update_filter] += 1
    
    def add_densification_stats_motion(self, viewspace_point_tensor, update_filter, motion_tensor, opacity_tensor=None, scaling_tensor=None):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        if self.ewa_prune:
            self.opacity_accum[update_filter] += opacity_tensor[update_filter].view(-1, 1)
            self.scaling_accum[update_filter] += scaling_tensor[update_filter].view(-1, 1)
        self.xyz_motion_accum[update_filter] += torch.norm(motion_tensor[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    