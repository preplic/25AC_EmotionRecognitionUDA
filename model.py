import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch
import math

class Model():
    def Generator(self, pixelda=False):
        return Feature()

    def Classifier(self, num_classes):
            return Predictor(num_classes)

class GradReverse(Function):
    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * (-1.0))

# 自注意力模块
# 使用更简单的自注意力实现
class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SelfAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x形状: [batch_size, input_dim]
        batch_size = x.size(0)
        # 将特征视为序列
        x_seq = x.view(batch_size, -1, 1)  # [batch_size, input_dim, 1]
        
        # 计算注意力权重
        attn_weights = self.attention(x)  # [batch_size, input_dim]
        
        # 应用注意力权重
        weighted_x = x * attn_weights  # [batch_size, input_dim]
        
        return weighted_x

# 通道注意力模块
# 修改 ChannelAttention 类
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x形状: [batch_size, in_channels]
        batch_size = x.size(0)
        
        # 直接使用特征作为通道
        # 计算特征的重要性
        out = self.fc(x)
        out = self.sigmoid(out)
        
        # 返回注意力权重乘以输入
        return x * out

# 频带注意力模块（将输入按频带分组）
# 修改 FrequencyBandAttention 类
class FrequencyBandAttention(nn.Module):
    def __init__(self, num_bands=5, band_features=32):
        super(FrequencyBandAttention, self).__init__()
        self.num_bands = num_bands
        self.band_features = band_features
        self.total_features = num_bands * band_features
        
        # 简化注意力机制
        self.attention = nn.Sequential(
            nn.Linear(self.total_features, num_bands),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x形状: [batch_size, features]
        batch_size = x.size(0)
        
        # 确保特征维度正确
        if x.size(1) != self.total_features:
            # 如果特征维度不匹配，不进行分频带处理，直接返回
            return x
            
        # 将特征重组为频带形式
        x_bands = x.view(batch_size, self.num_bands, self.band_features)
        
        # 计算频带的重要性权重
        weights = self.attention(x)  # [batch_size, num_bands]
        
        # 对每个频带应用权重
        x_weighted = x_bands * weights.unsqueeze(-1)  # [batch_size, num_bands, band_features]
        
        # 重新展平
        return x_weighted.view(batch_size, -1)

# 改进的特征提取器
class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        # 第一层网络
        self.fc1 = nn.Linear(160, 128)
        self.bn1_fc = nn.BatchNorm1d(128)
        
        # 注意力机制 - 在特征提取过程中应用
        # 自注意力，帮助捕捉不同时间点/特征之间的关系
        self.self_attention = SelfAttention(128, hidden_dim=64)
        
        # 通道注意力，赋予不同通道不同的重要性
        self.channel_attention = ChannelAttention(128, reduction_ratio=4)
        
        # 频带注意力，假设128个特征可以分为4个频带，每个频带有32个特征
        # 这里的参数需要根据实际数据的组织方式调整
        self.freq_attention = FrequencyBandAttention(num_bands=4, band_features=32)
        
        # 第二层网络
        self.fc2 = nn.Linear(128, 64)
        self.bn2_fc = nn.BatchNorm1d(64)
        
        # 输出前的注意力机制，增强最终特征的表示
        self.final_attention = SelfAttention(64, hidden_dim=32)
        
    # 修改 Feature 类的 forward 方法
    def forward(self, x, reverse=False):
        if reverse:
            x = GradReverse.apply(x)
        
        # 展平输入
        x = x.view(x.size(0), -1)  # 确保输入被完全展平为 [batch_size, features]
        
        # 第一层特征提取
        x = self.fc1(x)
        x = self.bn1_fc(x)
        x = F.relu(x)
        
        # 使用残差连接方式应用注意力
        identity = x
        
        # 应用自注意力
        x = self.self_attention(x) + identity
        
        # 应用通道注意力
        x = self.channel_attention(x) + x
        
        # 在可能的情况下应用频带注意力
        try:
            x = self.freq_attention(x) + x
        except:
            # 如果频带注意力不适用，则跳过
            pass
        
        # 第二层特征提取
        x = self.fc2(x)
        x = self.bn2_fc(x)
        x = F.relu(x)
        
        # 最终注意力
        x = self.final_attention(x)
        
        return x

# 分类器保持不变
class Predictor(nn.Module):
    def __init__(self, num_classes, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(64, 32)
        self.bn1_fc = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 10)
        self.bn2_fc = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, num_classes)
        self.bn_fc3 = nn.BatchNorm1d(num_classes)
        self.fc4 = nn.Softmax(dim=1)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = GradReverse.apply(x)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x_prev = self.fc3(x)
        x = self.fc4(x_prev)
        return x, x_prev
