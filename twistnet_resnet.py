# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import math
import numpy as np
import mmcv

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule

from mmpretrain.models.utils import find_nearest_divisible_factor
#kaiming_init, constant_init
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init, normal_init
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.logging.logger import MMLogger

# from mmpretrain.utils import get_root_logger

from mmpretrain.registry import MODELS
from mmpretrain.models.builder import BACKBONES
from mmpretrain.models.backbones.resnet import BasicBlock, Bottleneck, ResLayer

from mmpretrain.models.utils.make_divisible import make_divisible


class Fusion_Block(nn.Module):
    def __init__(self):
        super(Fusion_Block, self).__init__()
    def forward(self, T1_weight , T2_weight):
        fused_3d= torch.cat([T1_weight, T2_weight], dim=1)
        fused_2d = fused_3d.unbind(dim=2)#list中有13个(16,6,256,256)的tensor
        outs = [fused_2d, fused_3d]
        return outs

class M_Module(BaseModule):
    def __init__(self,
                 inplanes,
                 planes,
                 paths=4,
                 groups=4,
                 expand_ratio=4,
                 conv_cfg=dict(type="Conv3d"),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type="ReLU"),
                 order=('conv', 'norm', 'act'),
                 drop_prob=0.0,
                 with_cp=False,
                 init_cfg=None):
        super(M_Module, self).__init__(init_cfg)

        self.expand_ratio = expand_ratio
        self.paths = paths
        self.groups = groups
        self.with_cp = with_cp
        width = inplanes * expand_ratio // paths

        if self.expand_ratio > 1:
            self.expand_conv = ConvModule(
                inplanes,
                inplanes * (expand_ratio - 1),
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                order=order,
                groups=inplanes)

        self.convs = nn.ModuleList()
        self.dcns = nn.ModuleList()

        for i in range(paths):
            if i == 0:
                self.convs.append(nn.Identity())
                self.dcns.append(nn.Identity())
            else:
                self.convs.append(
                    ConvModule(
                        width,
                        width // 2,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        groups=groups,
                        order=order))
                self.dcns.append(
                    ConvModule(
                        width // 2,
                        width // 2,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        groups=width // 2,
                        order=order))

        self.compress_conv = \
            ConvModule(
                width * paths,
                planes,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                order=order)

        self.drop_path = nn.Dropout3d(drop_prob) if drop_prob > 0.0 else nn.Identity()

        if inplanes != planes:
            self.res = \
                ConvModule(
                    inplanes,
                    planes,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
        else:
            self.res = nn.Identity()

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):

            if self.expand_ratio > 1:
                out = torch.cat([x, self.expand_conv(x)], 1)
            else:
                out = x

            spx = torch.chunk(out, self.paths, 1)
            sp = self.convs[0](spx[0])
            sp_out = sp_outs = self.dcns[0](sp)
            for i in range(1, self.paths):
                sp = self.convs[i](spx[i] + sp_out)
                sp_out = self.dcns[i](sp) + sp
                sp_outs = torch.cat([sp_outs, sp_out], 1)

            out = self.compress_conv(sp_outs)

            out = self.drop_path(out)

            out = out + self.res(x)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out

class CT_Module(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 compress_ratio=4,
                 conv_cfg=dict(type="Conv3d"),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type="ReLU"),
                 order=('conv', 'norm', 'act'),
                 init_cfg=None
                 ):
        super(CT_Module, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_cfg = act_cfg
        self.compress_ratio = compress_ratio

        kernel_size = 1

        if compress_ratio == 1:
            if in_channels < out_channels:
                mid_channels = find_nearest_divisible_factor(in_channels, out_channels)
                self.expand_conv = \
                    ConvModule(
                        in_channels,
                        mid_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        order=order)
                self.cheap_conv = \
                    ConvModule(
                        mid_channels,
                        out_channels - mid_channels,
                        1,
                        groups=mid_channels,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        order=order)
            else:
                self.compress_conv = \
                    ConvModule(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        order=order)
        else:
            mid_channels = out_channels // compress_ratio
            if in_channels < out_channels:
                self.expand_conv = \
                    ConvModule(
                        in_channels,
                        mid_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        order=order)
            else:
                self.compress_conv = \
                    ConvModule(
                        in_channels,
                        mid_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        order=order)
            self.cheap_conv = \
                ConvModule(
                    mid_channels,
                    out_channels - mid_channels,
                    1,
                    groups=mid_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)

    def forward(self, x):
        if self.compress_ratio == 1:
            if self.in_channels < self.out_channels:
                out = self.expand_conv(x)
                out = torch.cat([out, self.cheap_conv(out)], 1)
            else:
                out = self.compress_conv(x)
        else:
            if self.in_channels < self.out_channels:
                out = self.expand_conv(x)
                out = torch.cat([out, self.cheap_conv(out)], 1)
            else:
                out = self.compress_conv(x)
                out = torch.cat([out, self.cheap_conv(out)], 1)

        return out

class Down_Layer(BaseModule):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=(1,2,2),
                 dilation=1,
                 conv_cfg=dict(type="Conv3d"),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type="ReLU"),
                 order=('conv', 'norm', 'act'),
                 drop_prob=0.0,
                 with_cp=False,
                 init_cfg=None):
        super(Down_Layer, self).__init__(init_cfg)


        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

        self.down_conv = \
            ConvModule(
                inplanes,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=(dilation, 1, 1),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                order=order)

        self.drop_path = nn.Dropout3d(drop_prob) if drop_prob > 0.0 else nn.Identity()

    def forward(self, x):
        def _inner_forward(x):

            out = self.down_conv(x)

            out = self.drop_path(out)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out

class Stage(BaseModule):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=(1,2,2),
                 dilation=1,
                 m_module_paths=4,
                 m_module_groups=4,
                 m_module_expand_ratio=4,
                 num_blocks=4,
                 conv_cfg=dict(type="Conv3d"),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type="ReLU"),
                 order=('conv', 'norm', 'act'),
                 drop_prob=0.0,
                 with_cp=False,
                 init_cfg=None):
        super(Stage, self).__init__(init_cfg)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.with_cp = with_cp

        self.down_layer = \
            Down_Layer(
                inplanes,
                planes,
                kernel_size=3,
                stride=stride,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                order=order,
                drop_prob=drop_prob,
                with_cp=with_cp,
                init_cfg=init_cfg)

        self.m_layer = nn.Sequential(*[
            M_Module(
                planes,
                planes,
                paths=m_module_paths,
                groups=m_module_groups,
                expand_ratio=m_module_expand_ratio,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                order=order,
                drop_prob=drop_prob,
                with_cp=with_cp,
                init_cfg=init_cfg) for _ in range(num_blocks)])

    def forward(self, x):
        def _inner_forward(x):
            x = self.down_layer(x)
            out = self.m_layer(x)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out

class BT_Block(BaseModule):
    def __init__(self,
                 slave_out_channels,
                 master_out_channels,
                 ct_module_compress_ratios=(4, 4),  # s2m, m2s
                 conv_cfg=dict(type="Conv3d"),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Sigmoid'),
                 bt_mode='dual_mul',
                 order=("conv", "norm", "act"),
                 drop_prob=-1,
                 with_cp=False,
                 init_cfg=None):
        super(BT_Block, self).__init__(init_cfg)
        assert bt_mode in ["dual_mul", "dual_add",
                           "s2m_mul", "s2m_add",
                           "m2s_mul", "m2s_add",
                           "dual_mul_add","dual_add_mul",
                           None,
                           "s2m_mul_e", "s2m_add_e",
                           "m2s_mul_e", "m2s_add_e",
                           "dual_mul_e", "dual_add_e",
                           "dual_mul_add_e","dual_add_mul_e"]

        self.bt_mode = bt_mode
        self.with_cp = with_cp
        self.resize_convs = nn.Identity()
        self.mix_conv = \
            ConvModule(
                slave_out_channels,
                slave_out_channels,
                (3,1,1),
                padding = (1,0,0),
                conv_cfg = conv_cfg,
                act_cfg = act_cfg,
                order = order)
        if bt_mode == "dual_mul" or bt_mode == "dual_add":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                    compress_ratio=ct_module_compress_ratios[0],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                    compress_ratio=ct_module_compress_ratios[1],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
        elif bt_mode == "dual_add_mul":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                    compress_ratio=ct_module_compress_ratios[0],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                    compress_ratio=ct_module_compress_ratios[1],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)

        elif bt_mode == "dual_mul_add":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                    compress_ratio=ct_module_compress_ratios[0],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                    compress_ratio=ct_module_compress_ratios[1],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)

        elif bt_mode == "s2m_mul" or bt_mode == "s2m_add":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                    compress_ratio=ct_module_compress_ratios[0],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)

        elif bt_mode == "m2s_mul" or bt_mode == "m2s_add":
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                    compress_ratio=ct_module_compress_ratios[0],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)

        elif bt_mode == "dual_mul_e" or bt_mode == "dual_add_e":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                    compress_ratio=ct_module_compress_ratios[0],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                    compress_ratio=ct_module_compress_ratios[1],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.unify_conv = \
                DepthwiseSeparableConvModule(
                    slave_out_channels,
                    master_out_channels,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.gather_convs = \
                DepthwiseSeparableConvModule(
                    master_out_channels,
                    master_out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)

        elif bt_mode == "dual_add_mul_e":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                    compress_ratio=ct_module_compress_ratios[0],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                    compress_ratio=ct_module_compress_ratios[1],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.unify_conv = \
                DepthwiseSeparableConvModule(
                    slave_out_channels,
                    master_out_channels,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.gather_convs = \
                DepthwiseSeparableConvModule(
                    master_out_channels,
                    master_out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)

        elif bt_mode == "dual_mul_add_e":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                    compress_ratio=ct_module_compress_ratios[0],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                    compress_ratio=ct_module_compress_ratios[1],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.unify_conv = \
                DepthwiseSeparableConvModule(
                    slave_out_channels,
                    master_out_channels,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.gather_convs = \
                DepthwiseSeparableConvModule(
                    master_out_channels,
                    master_out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)

        elif bt_mode == "s2m_mul_e" or bt_mode == "s2m_add_e":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                    compress_ratio=ct_module_compress_ratios[0],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='Sigmoid') if "mul" in bt_mode else act_cfg,
                    order=order)
            self.unify_conv = \
                DepthwiseSeparableConvModule(
                    slave_out_channels,
                    master_out_channels,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.gather_convs = \
                DepthwiseSeparableConvModule(
                    master_out_channels,
                    master_out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)

        elif bt_mode == "m2s_mul_e" or bt_mode == "m2s_add_e":
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                    compress_ratio=ct_module_compress_ratios[1],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='Sigmoid') if "mul" in bt_mode else act_cfg,
                    order=order)
            self.unify_conv = \
                DepthwiseSeparableConvModule(
                    slave_out_channels,
                    master_out_channels,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.gather_convs = \
                DepthwiseSeparableConvModule(
                    master_out_channels,
                    master_out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)

        else:
            self.s2m_conv = nn.Identity()
            self.m2s_conv = nn.Identity()

        self.drop_path = nn.Dropout3d(drop_prob) if drop_prob > 0.0 else nn.Identity()

    def forward(self, x):

        def _inner_forward(x):
            x_slave_list, x_master = x[0], x[1]
            bs, channel, weight, width = x_slave_list[0].shape
            x_slave_reshaped = [arr.reshape(bs, channel, 1, weight, width) for arr in x_slave_list]  # (16,6,1,256,256)
            x_slave = torch.cat(x_slave_reshaped, dim=2)
            batch_size, channels0, original_depth0, height0, width0 = x_slave.shape
            batch_size, channels1, original_depth1, height1, width1 = x_master.shape
            x_slave = nn.functional.interpolate(x_slave, size=(original_depth1, height1, width1),
                                                        mode='trilinear', align_corners=False)
            x_master = nn.functional.interpolate(x_master, size=(original_depth0, height0, width0),
                                                         mode='trilinear', align_corners=False)
            if self.bt_mode == "dual_mul" or self.bt_mode == "dual_add":
                out_master = self.s2m_conv(self.mix_conv(x_slave))
                out_master = self.drop_path(out_master)

                if "mul" in self.bt_mode:
                    out_master = x_master * out_master
                else:
                    out_master = x_master + out_master

                out_slave = self.m2s_conv(x_master)
                out_slave = self.drop_path(out_slave)
                if "mul" in self.bt_mode:
                    out_slave = x_slave * out_slave
                else:
                    out_slave = x_slave + out_slave
                out_slave = out_slave.unbind(dim=2) 
                results = [out_slave, out_master]

            elif self.bt_mode == "dual_add_mul":
                out_master = self.s2m_conv(self.mix_conv(x_slave))
                out_master = self.drop_path(out_master)
                out_master = x_master + out_master

                out_slave = self.m2s_conv(x_master)
                out_slave = self.drop_path(out_slave)
                out_slave = x_slave * out_slave
                out_slave = out_slave.unbind(dim=2) 
                results = [out_slave, out_master]

            elif self.bt_mode == "s2m_mul" or self.bt_mode == "s2m_add":
                out_master = self.s2m_conv(self.mix_conv(x_slave))
                if "mul" in self.bt_mode:
                    out_master = x_master * out_master
                else:
                    out_master = x_master + out_master
                out_master = self.drop_path(out_master)

                out_slave = self.drop_path(x_slave)
                out_slave = out_slave.unbind(dim=2)  

                results = [out_slave, out_master]

            elif self.bt_mode == "m2s_mul" or self.bt_mode == "m2s_add":
                out_slave = self.m2s_conv(x_master)
                if "mul" in self.bt_mode:
                    out_slave = x_slave * out_slave
                else:
                    out_slave = x_slave + out_slave
                out_slave = self.drop_path(out_slave)
                out_slave = out_slave.unbind(dim=2) 

                out_master = self.drop_path(x_master)

                results = [out_slave, out_master]

            elif self.bt_mode == "dual_mul_e" or self.bt_mode == "dual_add_e":
                out_master = self.s2m_conv(self.mix_conv(x_slave))
                out_master = self.drop_path(out_master)

                if "mul" in self.bt_mode:
                    out_master = x_master * out_master
                else:
                    out_master = x_master + out_master

                out_slave = self.m2s_conv(x_master)
                out_slave = self.drop_path(out_slave)
                if "mul" in self.bt_mode:
                    out_slave = x_slave * out_slave
                else:
                    out_slave = x_slave + out_slave

                out_slave = self.unify_conv(out_slave)
                out_slave = self.drop_path(out_slave)

                results = self.gather_convs(out_master + out_slave)


                results = [self.drop_path(results)]

            elif self.bt_mode == "dual_add_mul_e":
                out_master = self.s2m_conv(self.mix_conv(x_slave))
                out_master = self.drop_path(out_master)
                out_master = x_master + out_master

                out_slave = self.m2s_conv(x_master)
                out_slave = self.drop_path(out_slave)
                out_slave = x_slave * out_slave
                out_slave = self.unify_conv(out_slave)
                out_slave = self.drop_path(out_slave)

                results = self.gather_convs(out_master + out_slave)

                results = [self.drop_path(results)]

            elif self.bt_mode == "s2m_mul_e" or self.bt_mode == "s2m_add_e":
                out_master = self.s2m_conv(self.mix_conv(x_slave))
                out_master = self.drop_path(out_master)

                if "mul" in self.bt_mode:
                    out_master = x_master * out_master
                else:
                    out_master = x_master + out_master

                out_slave = self.unify_conv(x_slave)
                out_slave = self.drop_path(out_slave)

                results = self.gather_convs(out_master + out_slave)

                results = [self.drop_path(results)]

            elif self.bt_mode == "m2s_mul_e" or self.bt_mode == "m2s_add_e":

                out_slave = self.m2s_conv(x_master)
                out_slave = self.drop_path(out_slave)
                if "mul" in self.bt_mode:
                    out_slave = x_slave * out_slave
                else:
                    out_slave = x_slave + out_slave

                out_slave = self.unify_conv(out_slave)
                out_slave = self.drop_path(out_slave)

                results = self.gather_convs(x_master + out_slave)

                results = [self.drop_path(results)]

            else:
                results = [self.drop_path(self.s2m_conv(x_slave)),
                           self.drop_path(self.m2s_conv(x_master))]

            return results

        if self.with_cp and (x[0].requires_grad or x[1].requires_grad):
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)

        return outs

@MODELS.register_module()
class TwistNet_ResNet(BaseModule):
    # The Hybrid Bilateral Network
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 # ------------------Semantic Part----------------------
                 slave_depth,
                 slave_strides=(1, 2, 2, 2),
                 slave_dilations=(1, 1, 1, 1),
                 slave_avg_down=False,
                 slave_conv_cfg=None,
                 slave_norm_cfg=dict(type='BN', requires_grad=True),
                 slave_norm_eval=False,
                 slave_act_cfg=dict(type='ReLU'),
                 slave_drop_prob=-1,
                 slave_init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),

                 # --------------------Detail Part----------------------
                 master_channels=(32, 64, 128, 256),
                 master_blocks=(2, 2, 2, 2),
                 master_strides=[(1,2,2),(1,2,2),(1,2,2),(1,2,2)],
                 master_dilations=[(1,1,1),(2,1,1),(4,1,1),(8,1,1)],
                 m_module_paths=4,
                 m_module_groups=4,
                 m_module_expand_ratio=4,
                 ct_module_compress_ratios=(1, 1),  # s2m, m2s
                 bt_modes=('dual_mul_add', 'dual_mul_add', 'dual_mul_add', 'dual_mul_add_e'),
                 master_order=('conv', 'norm', 'act'),
                 master_conv_cfg=dict(type="Conv3d"),
                 master_norm_cfg=dict(type='BN', requires_grad=True),
                 master_norm_eval=False,
                 master_act_cfg=dict(type='ReLU'),
                 master_drop_prob=-1,

                 # ---------------------Common Part----------------------
                 with_cp=False,
                 multi_modals=2):
        super(TwistNet_ResNet, self).__init__(init_cfg=None)

        # -------------------Slave----------------------
        if slave_depth not in self.arch_settings:
            raise KeyError(f'invalid slave_depth {slave_depth} for resnet')
        assert len(slave_strides) == len(slave_dilations)
        stem_channels = 64
        self.slave_norm_eval = slave_norm_eval
        self.block, self.stage_blocks = self.arch_settings[slave_depth]
        self.slave_init_cfg = slave_init_cfg

        # --------------------Master----------------------
        assert len(master_channels) == len(master_blocks) \
               == len(master_strides) == len(slave_strides)
        temp = []
        for master_channel in master_channels:
            mid_planes = make_divisible(master_channel, np.lcm(m_module_paths, m_module_groups))
            temp.append(mid_planes)
        master_channels = temp

        self.master_norm_eval = master_norm_eval

        # ------------------Common----------------------
        self.with_cp = with_cp
        self.multi_modals = multi_modals

        # -------------------Slave-----------------------
        self.slave_stages = []

        # --------------------Master----------------------
        self.master_stages = []
        self.bt_stages = []

        self.slave_stem = nn.Sequential(
            ConvModule(
                multi_modals * 3,
                stem_channels ,
                kernel_size=7,
                stride=2,
                padding=3,
                conv_cfg=None,
                norm_cfg=slave_norm_cfg,
                act_cfg=dict(type='ReLU'),
                bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.master_stem = \
            ConvModule(
                multi_modals * 3,
                stem_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=dict(type="Conv3d"),
                norm_cfg=master_norm_cfg,
                act_cfg=dict(type='ReLU'),
                bias=False)

        _slave_inplanes = stem_channels
        _slave_planes = stem_channels * self.block.expansion
        for i in range(len(self.stage_blocks)):
            slave_stage = self.make_slave_stage(
                block=self.block,
                num_blocks=self.stage_blocks[i],
                inplanes=_slave_inplanes,
                planes=_slave_planes // self.block.expansion,
                stride=slave_strides[i],
                dilation=slave_dilations[i],
                style='pytorch',
                avg_down=slave_avg_down,
                with_cp=with_cp,
                conv_cfg=slave_conv_cfg,
                norm_cfg=slave_norm_cfg,
                contract_dilation=True,
                drop_prob=slave_drop_prob)
            stage_name = f'slave_stage_{i + 1}'
            self.add_module(stage_name, slave_stage)
            self.slave_stages.append(stage_name)

            master_stage = Stage(
                inplanes=master_channels[i - 1] if i != 0 else stem_channels // 2,
                planes=master_channels[i],
                stride=master_strides[i],
                dilation=master_dilations[i],
                m_module_paths=m_module_paths,
                m_module_groups=m_module_groups,
                m_module_expand_ratio=m_module_expand_ratio,
                num_blocks=master_blocks[i],
                conv_cfg=master_conv_cfg,
                norm_cfg=master_norm_cfg,
                act_cfg=master_act_cfg,
                order=master_order,
                drop_prob=master_drop_prob,
                with_cp=with_cp,
                init_cfg=None)
            master_stage_name = f'master_stage_{i + 1}'
            self.add_module(master_stage_name, master_stage)
            self.master_stages.append(master_stage_name)

            bt_stage = BT_Block(
                slave_out_channels=_slave_planes,
                master_out_channels=master_channels[i],
                ct_module_compress_ratios=ct_module_compress_ratios,
                conv_cfg=master_conv_cfg,
                norm_cfg=master_norm_cfg,
                act_cfg=master_act_cfg,
                bt_mode=bt_modes[i],
                order=master_order,
                drop_prob=master_drop_prob,
                with_cp=with_cp,
                init_cfg=None)
            bt_stage_name = f'master_bt_stage_{i + 1}'
            self.add_module(bt_stage_name, bt_stage)
            self.bt_stages.append(bt_stage_name)

            _slave_inplanes = _slave_planes
            _slave_planes *= 2

        self._freeze_stages()

    def make_slave_stage(self, **kwargs):
        return ResLayer(**kwargs)

    def _freeze_stages(self):
        pass

    def init_weights(self):
        super(TwistNet_ResNet, self).init_weights()
        # ------------------Semantic Part----------------------
        if self.slave_init_cfg is not None:
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, self.slave_init_cfg['checkpoint'], strict=False, logger=logger)
        else:
            for name, m in self.named_modules():
                if "stem" in name and "master" not in name:
                    if isinstance(m, nn.Conv2d):
                        normal_init(m, mean=0, std=0.01)
                    elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                        normal_init(m, mean=0, std=0.01)

        # --------------------Detail Part----------------------
        for name, m in self.named_modules():
            if "master" in name and "stem" not in name:
                if isinstance(m, nn.Conv3d):
                    normal_init(m, mean=0, std=0.01)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, val=1, bias=0.0001)

    def forward(self, x1, x2):
        # Fusion_Block(x1,x2)
        fusion_block = self.fusion_block(x1, x2)
        x_slave = fusion_block[0],
        x_master = fusion_block[1],

        for i in range(len(self.stage_blocks)):
            slave_name = self.slave_stages[i]
            slave_stage = getattr(self, slave_name)
            x_slave = slave_stage(x_slave)

            master_stage_name = self.master_stages[i]
            master_stage = getattr(self, master_stage_name)
            x_master = master_stage(x_master)

            bt_stage_name = self.bt_stages[i]
            bt_stage = getattr(self, bt_stage_name)
            bt_outs = bt_stage([x_slave, x_master])

            if len(bt_outs) > 1:
                x_slave, x_master = bt_outs[0], bt_outs[1]
            else:
                out = bt_outs[0]

        return tuple([out])

    def train(self, mode=True):
        super(TwistNet_ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
