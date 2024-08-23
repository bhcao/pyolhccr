import torch
from torch import nn
from torch.nn import functional as F

from pyolhccr.utils import Preprocess

class StrokeGNNLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(StrokeGNNLayer, self).__init__()

        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        return self.bn(x)
    
class PositionGNNLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(PositionGNNLayer, self).__init__()

        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        return self.bn(x)

class GNN(nn.Module):
    '''Reference: https://arxiv.org/abs/2111.03281v2 for two stream gnn
    '''
    transform = Preprocess(
        Preprocess.ToBezierCurves()
    )

    transform_plus = Preprocess(
        Preprocess.ToBezierCurves(),
        Preprocess.FeatureExtraction()
    )

    def __init__(self):
        super(GNN, self).__init__()

    def forward(self, x):
        return x
    