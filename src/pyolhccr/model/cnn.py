from torch import nn
from pyolhccr.utils import Preprocess

class CNN(nn.Module):
    transform = Preprocess(
        Preprocess.Rasterization()
    )

    transform_plus = Preprocess(
        Preprocess.Rasterization(),
        Preprocess.FeatureExtraction()
    )

    def __init__(self):
        super(CNN, self).__init__()

    def forward(self, x):
        return x