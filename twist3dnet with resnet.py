import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torchvision.models as models
from torchvision.models import ResNet18_Weights


def Conv(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )

def DepthWiseConv3D(inp, out):
    return nn.Sequential(
        nn.Conv3d(inp, inp, 3, 1, 1, groups=inp, bias=False),
        nn.BatchNorm3d(inp),
        nn.ReLU(inplace=True),
        nn.Conv3d(inp, out, 1, 1, 0, bias=False),
        nn.BatchNorm3d(out),
        nn.ReLU(inplace=True),
    )

def DConv(inp, out):
    return nn.Sequential(
        nn.Conv3d(inp, out, 1, 1, 0, bias=False),
        nn.BatchNorm3d(out),
        nn.ReLU(inplace=True),
    )


class Fusion_Block(nn.Module):
    def __init__(self):
        super(Fusion_Block, self).__init__()

    def forward(self, T1_weight, T2_weight):
        T1_weight = T1_weight.repeat(1, 3, 1, 1, 1)
        T2_weight = T2_weight.repeat(1, 3, 1, 1, 1)
        fused_3d = torch.cat([T1_weight, T2_weight], dim=1)
        batch_size, channels, depth, height, width = fused_3d.shape
        fused_2d = fused_3d.permute(0, 2, 1, 3, 4).contiguous().view(-1, channels, height, width)
        return fused_2d, fused_3d


class M_Module(nn.Module):
    def __init__(self, in_channels):
        super(M_Module, self).__init__()
        self.dconv = DConv(in_channels, in_channels)
        # k=4, g=4: in_channels//2 → in_channels//4, groups=4
        self.gconv = nn.Sequential(
            nn.Conv3d(in_channels // 2, in_channels // 4, kernel_size=3, groups=4, padding=1),
            nn.BatchNorm3d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dconv1 = DConv(in_channels // 4, in_channels // 4)
        self.dropout = nn.Dropout3d(p=0.3)

    def forward(self, x):
        y1 = self.dconv(x)
        y2 = torch.cat([x, y1], dim=1)               # 2C
        p0, p1, p2, p3 = torch.chunk(y2, 4, dim=1)   # k=4, each C/2
        q1 = p1 + p0                                  # C/2
        s1 = self.gconv(q1)                           # C/4
        t1 = self.dconv1(s1)                          # C/4
        s1 = self.dropout(s1)
        v1 = torch.cat([t1, s1], dim=1)               # C/2
        q2 = p2 + v1                                  # C/2
        s2 = self.gconv(q2)                           # C/4
        t2 = self.dconv1(s2)                          # C/4
        s2 = self.dropout(s2)
        v2 = torch.cat([t2, s2], dim=1)               # C/2
        q3 = p3 + v2                                  # C/2
        s3 = self.gconv(q3)                           # C/4
        t3 = self.dconv1(s3)                          # C/4
        s3 = self.dropout(s3)
        v3 = torch.cat([t3, s3], dim=1)               # C/2
        y3 = torch.cat([p0, v1, v2, v3], dim=1)       # 2C
        y4 = self.conv(y3)                            # C
        y4 = self.dropout(y4)
        output = self.relu(y4 + x)
        return output


def find_nearest_divisible_factor(value1, value2):
    factors = []
    for_times = int(math.sqrt(value2))
    for i in range(for_times + 1)[1:]:
        if value2 % i == 0:
            factors.append(i)
            t = int(value2 / i)
            if not t == i:
                factors.append(t)
    factors.sort()
    array = np.asarray(np.array(factors))
    idx = (np.abs(array - value1)).argmin()
    return array[idx]


class CT_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CT_Module, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = nn.Dropout3d(p=0.3)
        kernel_size = 1
        if in_channels < out_channels:
            mid_channels = find_nearest_divisible_factor(in_channels, out_channels)
            self.expand_conv = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.cheap_conv = DConv(mid_channels, out_channels - mid_channels)
        else:
            self.compress_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.in_channels < self.out_channels:
            out = self.expand_conv(x)
            out = self.dropout(out)
            out = torch.cat([out, self.cheap_conv(out)], 1)
        else:
            out = self.compress_conv(x)
            out = self.dropout(out)
        return out


class Down_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(Down_Layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = nn.Dropout3d(p=0.3)
        kernel_size = 3
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=(1, 2, 2), dilation=(dilation, 1, 1),
                      padding=(dilation, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.down_conv(x)
        out = self.dropout(out)
        return out


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, num_block=1):
        super(Stage, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = 3
        self.down_layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=(1, 2, 2), padding=(dilation, 1, 1),
                      dilation=(dilation, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.m_layer = nn.Sequential(*[M_Module(out_channels) for _ in range(num_block)])
        self.dropout = nn.Dropout3d(p=0.3)

    def forward(self, x):
        x = self.down_layer(x)
        x = self.dropout(x)
        out = self.m_layer(x)
        out = self.dropout(out)
        return out


class BT_Block(nn.Module):
    def __init__(self, slave_out_channels, master_out_channels,
                 bt_mode='dual_add_mul', drop_prob=0.3):
        super(BT_Block, self).__init__()
        self.bt_mode = bt_mode
        self.mix_conv = nn.Sequential(
            nn.Conv3d(slave_out_channels, slave_out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1),
                      padding=(1, 0, 0)),
            nn.BatchNorm3d(slave_out_channels),
            nn.ReLU(inplace=True)
        )
        if bt_mode in ("dual_mul", "dual_add", "dual_add_mul", "dual_mul_add"):
            self.s2m_conv = CT_Module(slave_out_channels, master_out_channels)
            self.m2s_conv = CT_Module(master_out_channels, slave_out_channels)
        elif bt_mode in ("s2m_mul", "s2m_add"):
            self.s2m_conv = CT_Module(slave_out_channels, master_out_channels)
        elif bt_mode in ("m2s_mul", "m2s_add"):
            self.m2s_conv = CT_Module(master_out_channels, slave_out_channels)
        elif bt_mode in ("dual_mul_e", "dual_add_e", "dual_mul_add_e", "dual_add_mul_e"):
            self.s2m_conv = CT_Module(slave_out_channels, master_out_channels)
            self.m2s_conv = CT_Module(master_out_channels, slave_out_channels)
            self.unify_conv = DConv(slave_out_channels, master_out_channels)
            self.gather_conv = DepthWiseConv3D(master_out_channels, master_out_channels)
        elif bt_mode in ("s2m_mul_e", "s2m_add_e"):
            self.s2m_conv = CT_Module(slave_out_channels, master_out_channels)
            self.gather_conv = DepthWiseConv3D(master_out_channels, master_out_channels)
        elif bt_mode in ("m2s_mul_e", "m2s_add_e"):
            self.m2s_conv = CT_Module(master_out_channels, slave_out_channels)
            self.unify_conv = DConv(slave_out_channels, master_out_channels)
            self.gather_conv = DepthWiseConv3D(master_out_channels, master_out_channels)
        else:
            self.s2m_conv = nn.Identity()
            self.m2s_conv = nn.Identity()
        self.drop_path = nn.Dropout3d(drop_prob) if drop_prob > 0.0 else nn.Identity()

    def forward(self, x):
        x_slave, x_master = x
        _, c1, h1, w1 = x_slave.shape
        bs2, c2, d2, h2, w2 = x_master.shape
        x_slave = x_slave.view(bs2, 13, c1, h1, w1).permute(0, 2, 1, 3, 4)
        x_slave2m = F.interpolate(x_slave, size=(d2, h2, w2), mode='trilinear', align_corners=False)
        x_master2s = F.interpolate(x_master, size=(13, h1, w1), mode='trilinear', align_corners=False)

        if self.bt_mode in ("dual_mul", "dual_add"):
            o0 = self.drop_path(self.s2m_conv(self.mix_conv(x_slave2m)))
            out_master = x_master * torch.sigmoid(o0) if "mul" in self.bt_mode else x_master + o0
            o1 = self.m2s_conv(x_master2s)
            out_slave = x_slave * torch.sigmoid(o1) if "mul" in self.bt_mode else x_slave + o1
            out_slave = out_slave.permute(0, 2, 1, 3, 4).contiguous().view(-1, c1, h1, w1)
            return out_slave, out_master
        elif self.bt_mode in ("dual_add_mul", "dual_mul_add"):
            o0 = self.drop_path(self.s2m_conv(self.mix_conv(x_slave2m)))
            out_master = x_master * torch.sigmoid(o0) + x_master if "mul_add" in self.bt_mode else x_master + o0
            o1 = self.m2s_conv(x_master2s)
            out_slave = x_slave * torch.sigmoid(o1) if "mul" in self.bt_mode else x_slave + o1
            out_slave = out_slave.permute(0, 2, 1, 3, 4).contiguous().view(-1, c1, h1, w1)
            return out_slave, out_master
        elif self.bt_mode in ("s2m_mul", "s2m_add"):
            o0 = self.s2m_conv(self.mix_conv(x_slave2m))
            out_master = x_master * torch.sigmoid(o0) if "mul" in self.bt_mode else x_master + o0
            out_master = self.drop_path(out_master)
            out_slave = self.drop_path(x_slave).permute(0, 2, 1, 3, 4).contiguous().view(-1, c1, h1, w1)
            return out_slave, out_master
        elif self.bt_mode in ("m2s_mul", "m2s_add"):
            o1 = self.m2s_conv(x_master2s)
            out_slave = x_slave * torch.sigmoid(o1) if "mul" in self.bt_mode else x_slave + o1
            out_slave = self.drop_path(out_slave).permute(0, 2, 1, 3, 4).contiguous().view(-1, c1, h1, w1)
            out_master = self.drop_path(x_master)
            return out_slave, out_master
        elif self.bt_mode in ("dual_mul_e", "dual_add_e", "dual_mul_add_e", "dual_add_mul_e"):
            o0 = self.drop_path(self.s2m_conv(self.mix_conv(x_slave2m)))
            out_master = x_master * torch.sigmoid(o0) if "mul" in self.bt_mode else x_master + o0
            if "add" in self.bt_mode.replace("mul", "").replace("_e", ""):
                out_master = x_master + out_master
            o1 = self.m2s_conv(x_master2s)
            out_slave = x_slave * torch.sigmoid(o1) if "mul" in self.bt_mode else x_slave + o1
            out_slave = F.interpolate(out_slave, size=(d2, h2, w2), mode='trilinear', align_corners=False)
            out_slave = self.unify_conv(out_slave)
            out_slave = self.drop_path(out_slave)
            out = self.gather_conv(out_master + out_slave)
            return self.drop_path(out)
        elif self.bt_mode in ("s2m_mul_e", "s2m_add_e"):
            o0 = self.drop_path(self.s2m_conv(self.mix_conv(x_slave2m)))
            out_master = x_master * torch.sigmoid(o0) if "mul" in self.bt_mode else x_master + o0
            out_slave = F.interpolate(x_master2s, size=(d2, h2, w2), mode='trilinear', align_corners=False)
            out_slave = self.drop_path(out_slave)
            out = self.gather_conv(out_master + out_slave)
            return self.drop_path(out)
        elif self.bt_mode in ("m2s_mul_e", "m2s_add_e"):
            o1 = self.drop_path(self.m2s_conv(x_master2s))
            out_slave = x_slave * torch.sigmoid(o1) if "mul" in self.bt_mode else x_slave + o1
            out_slave = F.interpolate(out_slave, size=(d2, h2, w2), mode='trilinear', align_corners=False)
            out_slave = self.unify_conv(out_slave)
            out_slave = self.drop_path(out_slave)
            out = self.gather_conv(x_master + out_slave)
            return self.drop_path(out)
        else:
            return self.drop_path(self.s2m_conv(x_slave)), self.drop_path(self.m2s_conv(x_master))


class Twist_ResNet_3D(nn.Module):
    def __init__(self, branch1_channels=(64, 128, 256, 512), branch2_channels=(32, 64, 128, 256),
                 branch2_dilations=(1, 2, 4, 8),
                 bt_modes=('dual_add_mul', 'dual_add_mul', 'dual_add_mul', 'dual_add_mul_e'),
                 multi_modals=2, num_classes=2, stem_channels=64):
        super(Twist_ResNet_3D, self).__init__()
        self.fusion_block = Fusion_Block()
        self.dropout = nn.Dropout3d(p=0.3)

        resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.branch1_stem = nn.Sequential(
            nn.Conv2d(multi_modals * 3, 64, kernel_size=7, stride=2, padding=3),
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool
        )
        self.branch1_layer1 = resnet18.layer1
        self.branch1_layer2 = resnet18.layer2
        self.branch1_layer3 = resnet18.layer3
        self.branch1_layer4 = resnet18.layer4

        self.branch2_stem = nn.Sequential(
            nn.Conv3d(multi_modals * 3, stem_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(stem_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.branch2_layer1 = Stage(branch2_channels[0], branch2_channels[0], dilation=branch2_dilations[0])
        self.branch2_layer2 = Stage(branch2_channels[0], branch2_channels[1], dilation=branch2_dilations[1])
        self.branch2_layer3 = Stage(branch2_channels[1], branch2_channels[2], dilation=branch2_dilations[2])
        self.branch2_layer4 = Stage(branch2_channels[2], branch2_channels[3], dilation=branch2_dilations[3])

        self.bt_module_1 = BT_Block(branch1_channels[0], branch2_channels[0], bt_mode=bt_modes[0])
        self.bt_module_2 = BT_Block(branch1_channels[1], branch2_channels[1], bt_mode=bt_modes[1])
        self.bt_module_3 = BT_Block(branch1_channels[2], branch2_channels[2], bt_mode=bt_modes[2])
        self.bg_module    = BT_Block(branch1_channels[3], branch2_channels[3], bt_mode=bt_modes[3])

        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(branch2_channels[3], num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        resnet18_pretrained = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        conv1_weight = resnet18_pretrained.conv1.weight.data
        new_conv1_weight = torch.cat([conv1_weight] * 2, dim=1) / 2
        self.branch1_stem[0].weight.data = new_conv1_weight
        self.branch1_stem[1].weight.data = resnet18_pretrained.bn1.weight.data
        self.branch1_stem[1].bias.data   = resnet18_pretrained.bn1.bias.data

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 2, D, H, W) — 通道0=T1_ROI, 通道1=T2_ROI
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        out1, out2 = self.fusion_block(t1, t2)
        out1 = self.branch1_stem(out1); out2 = self.branch2_stem(out2)
        out1 = self.branch1_layer1(out1); out2 = self.branch2_layer1(out2)
        out1, out2 = self.bt_module_1((out1, out2))
        out1 = self.branch1_layer2(out1); out2 = self.branch2_layer2(out2)
        out1, out2 = self.bt_module_2((out1, out2))
        out1 = self.branch1_layer3(out1); out2 = self.branch2_layer3(out2)
        out1, out2 = self.bt_module_3((out1, out2))
        out1 = self.branch1_layer4(out1); out2 = self.branch2_layer4(out2)
        out = self.bg_module((out1, out2))
        out = self.dropout(out); out = self.gap(out)
        out = torch.flatten(out, 1)
        return self.fc(out)
