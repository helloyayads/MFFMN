import torch
import torch.nn as nn

class ChannelSpatialAttentionFusion(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        # 使用 1x1 卷积生成自适应权重
        self.conv = nn.Conv2d(2 * channels, channels, kernel_size=1)
        
    def forward(self, x1, x2):
        # 拼接特征
        x_cat = torch.cat([x1, x2], dim=1)
        # 生成权重 alpha，范围 [0, 1]
        alpha = torch.sigmoid(self.conv(x_cat))
        # 加权融合
        return alpha * x1 + (1 - alpha) * x2

# # 示例用法
# x1 = torch.randn(2, 512, 400, 352)
# x2 = torch.randn(2, 512, 400, 352)
# fusion = ChannelSpatialAttentionFusion()
# output = fusion(x1, x2)
# print(output.shape)  # [2, 512, 400, 352]