import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class DenseLayer2D(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck_ratio=4, dropout_rate=0.0):
        super().__init__()
        inter_channels = growth_rate * bottleneck_ratio
        self.bn1 = nn.BatchNorm2d(in_channels); self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels); self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, 3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x))); out = self.dropout(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        return torch.cat([x, self.dropout(out)], dim=1)


class DenseBlock2D(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate, bottleneck_ratio=4, dropout_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList(); channels = in_channels
        for _ in range(num_layers):
            self.layers.append(DenseLayer2D(channels, growth_rate, bottleneck_ratio, dropout_rate))
            channels += growth_rate
        self.out_channels = channels
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x


class TransitionBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels=None, compression=0.5, dropout_rate=0.0):
        super().__init__()
        if out_channels is None: out_channels = int(in_channels * compression)
        self.bn = nn.BatchNorm2d(in_channels); self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.pool = nn.AvgPool2d(2, 2); self.out_channels = out_channels
    def forward(self, x): return self.pool(self.dropout(self.conv(self.relu(self.bn(x)))))


class DenseLayer3D(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck_ratio=4, dropout_rate=0.0):
        super().__init__()
        inter_channels = growth_rate * bottleneck_ratio
        self.bn1 = nn.BatchNorm3d(in_channels); self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, inter_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(inter_channels); self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(inter_channels, growth_rate, 3, padding=1, bias=False)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x))); out = self.dropout(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        return torch.cat([x, self.dropout(out)], dim=1)


class DenseBlock3D(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate, bottleneck_ratio=4, dropout_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList(); channels = in_channels
        for _ in range(num_layers):
            self.layers.append(DenseLayer3D(channels, growth_rate, bottleneck_ratio, dropout_rate))
            channels += growth_rate
        self.out_channels = channels
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x


class TransitionBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, compression=0.5, dropout_rate=0.0):
        super().__init__()
        if out_channels is None: out_channels = int(in_channels * compression)
        self.bn = nn.BatchNorm3d(in_channels); self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.pool = nn.AvgPool3d((1, 2, 2), (1, 2, 2), ceil_mode=True)
        self.out_channels = out_channels
    def forward(self, x): return self.pool(self.dropout(self.conv(self.relu(self.bn(x)))))


class DenseNet2DEncoder(nn.Module):
    def __init__(self, in_channels=2, initial_channels=128, growth_rate=48,
                 num_layers=[6,8,12,8], compression=0.5, dropout_rate=0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(initial_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        channels = initial_channels; self.blocks = nn.ModuleList(); self.transitions = nn.ModuleList()
        for i in range(3):
            block = DenseBlock2D(channels, num_layers[i], growth_rate, dropout_rate=dropout_rate)
            self.blocks.append(block); channels = block.out_channels
            t = TransitionBlock2D(channels, compression=compression, dropout_rate=dropout_rate)
            self.transitions.append(t); channels = t.out_channels
        final = DenseBlock2D(channels, num_layers[-1], growth_rate, dropout_rate=dropout_rate)
        self.blocks.append(final); self.final_channels = final.out_channels
        self.final_bn = nn.BatchNorm2d(self.final_channels); self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.stem(x)
        for block, trans in zip(self.blocks[:-1], self.transitions):
            x = block(x); x = trans(x)
        x = self.blocks[-1](x)
        return self.final_relu(self.final_bn(x)), []


class DenseNet3DEncoder(nn.Module):
    def __init__(self, in_channels=5, initial_channels=128, growth_rate=40,
                 num_layers=[3,4,8,6], compression=0.5, dropout_rate=0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, initial_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(initial_channels), nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1))
        channels = initial_channels; self.blocks = nn.ModuleList(); self.transitions = nn.ModuleList()
        for i in range(3):
            block = DenseBlock3D(channels, num_layers[i], growth_rate, dropout_rate=dropout_rate)
            self.blocks.append(block); channels = block.out_channels
            t = TransitionBlock3D(channels, compression=compression, dropout_rate=dropout_rate)
            self.transitions.append(t); channels = t.out_channels
        final = DenseBlock3D(channels, num_layers[-1], growth_rate, dropout_rate=dropout_rate)
        self.blocks.append(final); self.final_channels = final.out_channels
        self.final_bn = nn.BatchNorm3d(self.final_channels); self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.stem(x)
        for block, trans in zip(self.blocks[:-1], self.transitions):
            x = block(x); x = trans(x)
        x = self.blocks[-1](x)
        return self.final_relu(self.final_bn(x))


class HDenseClassifier(nn.Module):
    def __init__(self, in_channels=2, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.encoder_2d = DenseNet2DEncoder(in_channels=in_channels, dropout_rate=dropout_rate)
        d2_final = self.encoder_2d.final_channels
        self.seg_head_2d = nn.Sequential(
            nn.Conv2d(d2_final, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1))
        d3_in = in_channels + 3
        self.encoder_3d = DenseNet3DEncoder(in_channels=d3_in, dropout_rate=dropout_rate)
        d3_final = self.encoder_3d.final_channels; fc = 128
        self.proj_2d_to_3d = nn.Sequential(nn.Conv3d(d2_final, fc, 1), nn.BatchNorm3d(fc), nn.ReLU(inplace=True))
        self.proj_3d = nn.Sequential(nn.Conv3d(d3_final, fc, 1), nn.BatchNorm3d(fc), nn.ReLU(inplace=True))
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(fc*2, fc, 3, padding=1), nn.BatchNorm3d(fc), nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(dropout_rate),
            nn.Linear(fc, 512), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_2d = x.permute(0,2,1,3,4).reshape(B*D, C, H, W)
        d2_features, _ = self.encoder_2d(x_2d)
        seg_2d = self.seg_head_2d(d2_features)
        _, sc, sh, sw = seg_2d.shape
        seg_3d = seg_2d.reshape(B, D, sc, sh, sw).permute(0,2,1,3,4)
        seg_3d = F.interpolate(seg_3d, size=(D, H, W), mode='trilinear', align_corners=False)
        x_3d = torch.cat([x, seg_3d], dim=1)
        d3_features = self.encoder_3d(x_3d)
        _, _, de, he, we = d3_features.shape
        d2_3d = d2_features.reshape(B, D, -1, he, we).permute(0,2,1,3,4)
        d2_3d = F.interpolate(d2_3d, size=(de, he, we), mode='trilinear', align_corners=False)
        p2 = self.proj_2d_to_3d(d2_3d); p3 = self.proj_3d(d3_features)
        fused = self.fusion_conv(torch.cat([p2, p3], dim=1))
        return self.classifier(fused)
