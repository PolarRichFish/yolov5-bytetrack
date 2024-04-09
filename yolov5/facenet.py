from FNet import InceptionResnetV1
from torch import nn
import torch
from torch.nn import functional as F


class Arc(nn.Module):
    def __init__(self, feature_dim=2, cls_dim=10):
        super(Arc, self).__init__()
        self.W = nn.Parameter(torch.randn(feature_dim, cls_dim))

    def forward(self, feature, m=1, s=10):
        # x为 N V结构 在v上做二范数归一化
        x = F.normalize(feature, dim=1)
        # w 为 2  cls 结构 ， 在 feature上做二范数归一化
        w = F.normalize(self.W, dim=0)
        # 矩阵相乘  /10是为了防止梯度爆炸
        cos = torch.matmul(x, w) / 10

        a = torch.acos(cos)
        top = torch.exp(s*torch.cos(a + m))
        down2 = torch.sum(torch.exp(s*torch.cos(a)), dim=1, keepdim=True) - torch.exp(s*torch.cos(a))
        out = torch.log(top / (top + down2))
        return out


class MoudleNet(nn.Module):
    def __init__(self, cls_num):
        super(MoudleNet, self).__init__()
        self.cls_num = cls_num

        # self.back_bone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.back_bone = InceptionResnetV1(pretrained='../weights/vggface2.pt')
        # self.feature_layer = nn.Sequential(
        #     nn.BatchNorm1d(8631),
        #     nn.LeakyReLU(),
        #     nn.Linear(8631, 512)
        # )
        self.out_layer = Arc(512, self.cls_num)

    def forward(self, x):
        # 冻结主干网络的权重
        with torch.no_grad():
            out1 = self.back_bone(x)

        # feature = self.feature_layer(out1)
        # out = self.out_layer(feature)
        out = self.out_layer(out1)
        return out

    def get_feature(self, x):
        # out1 = self.back_bone(x)
        feature = self.back_bone(x)
        # feature = self.feature_layer(out1)
        return feature