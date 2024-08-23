from torch import nn
from pyolhccr.utils import Preprocess

class LSTM(nn.Module):
    transform = Preprocess(
        Preprocess.ToBezierCurves()
    )

    transform_plus = Preprocess(
        Preprocess.ToBezierCurves(),
        Preprocess.FeatureExtraction()
    )

    def __init__(self):
        super(LSTM, self).__init__()

    def forward(self, x):
        return x