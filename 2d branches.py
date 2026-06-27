# ShuffleNetV2 / SENet18 / ConvMixer

import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ShuffleNet_V2_X1_0_Weights



# 1. ShuffleNetV2 x1.0 


def build_shufflenetv2_branch(multi_modals=2):
    sf = models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    # sf.conv1: Conv2d(3→24,s2)+BN+ReLU, maxpool
    # sf.stage2: 24→116ch, /2 spatial
    # sf.stage3: 116→232ch, /2 spatial
    # sf.stage4: 232→464ch, /2 spatial
    # sf.conv5 : 464→1024ch, 1×1

    stem = nn.Sequential(
        nn.Conv2d(multi_modals * 3, 24, 3, stride=2, padding=1, bias=False),
        sf.conv1[1], sf.conv1[2], sf.maxpool)
    layer1 = sf.stage2
    layer2 = sf.stage3
    layer3 = sf.stage4
    layer4 = sf.conv5
    branch1_channels = (116, 232, 464, 1024)

    cw = sf.conv1[0].weight.data
    stem[0].weight.data = torch.cat([cw] * multi_modals, dim=1) / multi_modals

    return stem, layer1, layer2, layer3, layer4, branch1_channels



# 2. SENet18 2D 


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid())
    def forward(self, x): return x * self.se(x)


class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None: identity = self.downsample(x)
        return self.relu(out + identity)


def _make_layer(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride, bias=False),
            nn.BatchNorm2d(planes))
    layers = [block(inplanes, planes, stride, downsample)]
    inplanes = planes * block.expansion
    for _ in range(1, blocks): layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)


def _copy_se_weights(se_layer, rn_layer):
    for se, rn in zip(se_layer, rn_layer):
        se.conv1.weight.data.copy_(rn.conv1.weight.data)
        se.bn1.weight.data.copy_(rn.bn1.weight.data)
        se.bn1.bias.data.copy_(rn.bn1.bias.data)
        se.bn1.running_mean.copy_(rn.bn1.running_mean)
        se.bn1.running_var.copy_(rn.bn1.running_var)
        se.conv2.weight.data.copy_(rn.conv2.weight.data)
        se.bn2.weight.data.copy_(rn.bn2.weight.data)
        se.bn2.bias.data.copy_(rn.bn2.bias.data)
        se.bn2.running_mean.copy_(rn.bn2.running_mean)
        se.bn2.running_var.copy_(rn.bn2.running_var)
        if se.downsample is not None and rn.downsample is not None:
            se.downsample[0].weight.data.copy_(rn.downsample[0].weight.data)
            se.downsample[1].weight.data.copy_(rn.downsample[1].weight.data)
            se.downsample[1].bias.data.copy_(rn.downsample[1].bias.data)
            se.downsample[1].running_mean.copy_(rn.downsample[1].running_mean)
            se.downsample[1].running_var.copy_(rn.downsample[1].running_var)


def build_senet18_branch(multi_modals=2):
    rn18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    stem = nn.Sequential(
        nn.Conv2d(multi_modals * 3, 64, 7, stride=2, padding=3, bias=False),
        rn18.bn1, rn18.relu, rn18.maxpool)

    layer1 = _make_layer(SEBasicBlock, 64, 64, 2, stride=1)
    layer2 = _make_layer(SEBasicBlock, 64, 128, 2, stride=2)
    layer3 = _make_layer(SEBasicBlock, 128, 256, 2, stride=2)
    layer4 = _make_layer(SEBasicBlock, 256, 512, 2, stride=2)

    _copy_se_weights(layer1, rn18.layer1)
    _copy_se_weights(layer2, rn18.layer2)
    _copy_se_weights(layer3, rn18.layer3)
    _copy_se_weights(layer4, rn18.layer4)

    # conv1：3ch → multi_modals*3ch
    cw = rn18.conv1.weight.data
    stem[0].weight.data = torch.cat([cw] * multi_modals, dim=1) / multi_modals

    branch1_channels = (64, 128, 256, 512)
    return stem, layer1, layer2, layer3, layer4, branch1_channels



# 3. ConvMixer 2D 


class ConvMixerBlock(nn.Module):
    """深度卷积残差 + 1×1 卷积残差"""
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dw_norm = nn.BatchNorm2d(dim)
        self.dw_act  = nn.GELU()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.pw_norm = nn.BatchNorm2d(dim)
        self.pw_act  = nn.GELU()
        self.pw_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = x + self.dw_conv(self.dw_act(self.dw_norm(x)))
        x = x + self.pw_conv(self.pw_act(self.pw_norm(x)))
        return x


def build_convmixer_branch(multi_modals=2, depth=2, kernel_size=7):
    stem = nn.Sequential(
        nn.Conv2d(multi_modals * 3, 64, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64), nn.GELU(),
        nn.MaxPool2d(3, stride=2, padding=1))

    layer1 = nn.Sequential(*[ConvMixerBlock(64, kernel_size) for _ in range(depth)])
    layer2 = nn.Sequential(
        nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128), nn.GELU(),
        *[ConvMixerBlock(128, kernel_size) for _ in range(depth)])
    layer3 = nn.Sequential(
        nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256), nn.GELU(),
        *[ConvMixerBlock(256, kernel_size) for _ in range(depth)])
    layer4 = nn.Sequential(
        nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512), nn.GELU(),
        *[ConvMixerBlock(512, kernel_size) for _ in range(depth)])

    branch1_channels = (64, 128, 256, 512)
    return stem, layer1, layer2, layer3, layer4, branch1_channels
