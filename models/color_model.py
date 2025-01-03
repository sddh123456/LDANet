import torch
import torch.nn as nn
from .lda_model import ColorLDAModel
import torch.nn.functional as F


class MultiScaleChannelAttention(nn.Module):
    def __init__(self, in_planes, ratios=[1, 2, 4], reduction=16):
        super(MultiScaleChannelAttention, self).__init__()
        self.pools = nn.ModuleList()
        for r in ratios:
            self.pools.append(nn.AdaptiveAvgPool2d(r))
            self.pools.append(nn.AdaptiveMaxPool2d(r))
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes * len(self.pools), in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pooled = [pool(x) for pool in self.pools]
        # 上采样并拼接
        pooled = [F.interpolate(p, size=x.size()[2:], mode='nearest') for p in pooled]
        pooled = torch.cat(pooled, dim=1)
        out = self.fc(pooled)
        out = self.sigmoid(out)
        return out 


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x_cat)
        return self.sigmoid(x)


class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()
        # 1x1卷积支路
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1卷积后接3x3卷积支路
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1卷积后接5x5卷积支路
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )

        # 3x3最大池化后接1x1卷积支路
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

        # CBAM注意力模块
        self.ca = MultiScaleChannelAttention(ch1x1 + ch3x3 + ch5x5 + pool_proj)
        self.sa = SpatialAttention()

        # 残差连接
        self.shortcut = nn.Sequential()
        if in_channels != (ch1x1 + ch3x3 + ch5x5 + pool_proj):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, ch1x1 + ch3x3 + ch5x5 + pool_proj,
                          kernel_size=1),
                nn.BatchNorm2d(ch1x1 + ch3x3 + ch5x5 + pool_proj)
            )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # 将所有支路的输出在通道维度上拼接
        outputs = torch.cat([branch1, branch2, branch3, branch4], 1)

        # 应用CBAM注意力机制
        out = self.ca(outputs) * outputs
        out = self.sa(out) * out

        # 残差连接
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ColorHarmonyNet(nn.Module):
    def __init__(self, config, lda_model):
        super(ColorHarmonyNet, self).__init__()
        self.config = config
        self.lda_model = lda_model  # 预训练的 LDA 模型
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Inception模块
        self.inception_a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception_b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception_c = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception_d = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception_e = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception_f = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception_g = InceptionModule(528, 256, 160, 320, 32, 128, 128)

        # 特征池化
        self.feature_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 对齐特征维度到 256
        self.conv_after_pool = nn.Conv2d(832, 256, kernel_size=1, bias=False)

        # LDA 特征编码器
        self.lda_encoder = nn.Sequential(
            nn.Linear(config.NUM_TOPICS, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )

        # 特征融合和预测
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),  # 256（深度特征）+ 256（LDA 特征）= 512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, color_hist=None):
        device = x.device
        if x.size(2) < 75 or x.size(3) < 75:
            x = F.interpolate(x, size=(75, 75), mode='bilinear', align_corners=False)
        x = x.to(device)

        # 特征提取路径
        features = self.pre_layers(x)
        features = self.inception_a(features)
        features = self.inception_b(features)
        features = self.maxpool(features)
        features = self.inception_c(features)
        features = self.inception_d(features)
        features = self.inception_e(features)
        features = self.inception_f(features)
        features = self.inception_g(features)

        # 池化和特征处理
        outfeatures = self.feature_pool(features)
        features = self.conv_after_pool(outfeatures)
        deep_features = torch.flatten(features, 1)

        # LDA 特征提取
        if color_hist is not None:
            lda_topics = self.lda_model.transform([color_hist.cpu().numpy()])
            lda_features = self.lda_encoder(torch.FloatTensor(lda_topics).to(x.device))
        else:
            lda_features = torch.zeros(x.size(0), 256).to(x.device)

        # 特征融合
        combined_features = torch.cat([deep_features, lda_features], dim=1)
        harmony_score = self.fusion(combined_features)

        return harmony_score, outfeatures, lda_features, lda_topics