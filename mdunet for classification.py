import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, growth_rate):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False),
            nn.BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, growth_rate, 3, padding=1, bias=False))
    def forward(self, x): return torch.cat([x, self.layer(x)], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, growth_rate, num_layers=4):
        super().__init__()
        layers, channels = [], in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(channels, bottleneck_channels, growth_rate))
            channels += growth_rate
        self.block = nn.Sequential(*layers); self.out_channels = channels
    def forward(self, x): return self.block(x)


class MDUEncoder2D(nn.Module):
    def __init__(self, in_slices=5, base_channels=48):
        super().__init__()
        b = base_channels
        self.conv1 = ConvBNReLU(in_slices, b, 3, 1, 1)
        self.dense1 = DenseBlock(b, b, b//4, 4); self.pool1 = nn.MaxPool2d(2,2)
        self.dense2 = DenseBlock(self.dense1.out_channels, b*2, b//2, 4); self.pool2 = nn.MaxPool2d(2,2)
        self.dense3 = DenseBlock(self.dense2.out_channels, b*4, b, 4); self.pool3 = nn.MaxPool2d(2,2)
        self.dense4 = DenseBlock(self.dense3.out_channels, b*8, b*2, 4); self.pool4 = nn.MaxPool2d(2,2)
        self.bridge = DenseBlock(self.dense4.out_channels, b*16, b*4, 4)
        self.out_channels = self.bridge.out_channels

    def forward(self, x):
        x = self.conv1(x); x = self.pool1(self.dense1(x))
        x = self.pool2(self.dense2(x)); x = self.pool3(self.dense3(x))
        x = self.pool4(self.dense4(x)); return self.bridge(x)


class SliceAttentionPool(nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.Tanh(), nn.Linear(hidden_channels, 1))
    def forward(self, features):
        attn_weights = torch.softmax(self.attention(features), dim=1)
        return torch.sum(features * attn_weights, dim=1)


class MDUBinaryClsNet(nn.Module):
    def __init__(self, in_slices=5, base_channels=48, num_stacks=12,
                 classifier_hidden=1024, dropout=0.5, use_attention=True):
        super().__init__()
        self.in_slices = in_slices; self.num_stacks = num_stacks; self.use_attention = use_attention
        self.encoder = MDUEncoder2D(in_slices, base_channels)
        feature_channels = self.encoder.out_channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.volume_pool = SliceAttentionPool(feature_channels, 128) if use_attention else None
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_channels), nn.Dropout(dropout),
            nn.Linear(feature_channels, classifier_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(classifier_hidden, 1))

    def _make_25d_stacks(self, x):
        B, C, D, H, W = x.shape
        K, N, half = self.in_slices, self.num_stacks, self.in_slices // 2
        x = x[:, 0]
        centers = torch.linspace(0, D-1, steps=N, device=x.device).round().long()
        offsets = torch.arange(-half, half+1, device=x.device).long()
        indices = (centers[:, None] + offsets[None, :]).clamp(min=0, max=D-1)
        stacks = torch.index_select(x, dim=1, index=indices.reshape(-1))
        return stacks.view(B, N, K, H, W).view(B*N, K, H, W)

    def _encode_25d(self, x):
        return self.global_pool(self.encoder(x)).flatten(1)

    def forward(self, x):
        B = x.shape[0]; N = self.num_stacks
        stacks = self._make_25d_stacks(x)
        features = self._encode_25d(stacks).view(B, N, -1)
        volume_feature = self.volume_pool(features) if self.use_attention else features.mean(dim=1)
        return self.classifier(volume_feature).squeeze(1)


class MDUBinaryClsNetWrapper(nn.Module):
    """双通道输入 + 双类输出 包装"""
    def __init__(self, in_channels=2, num_classes=2, classifier_hidden=1024, **kwargs):
        super().__init__()
        self.input_fusion = nn.Conv3d(in_channels, 1, 1, bias=False)
        kwargs['classifier_hidden'] = classifier_hidden
        self.model = MDUBinaryClsNet(**kwargs)
        self.model.classifier[-1] = nn.Linear(classifier_hidden, num_classes)

    def forward(self, x):
        x = self.input_fusion(x)
        return self.model(x)
