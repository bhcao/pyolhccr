from torch import nn
from pyolhccr.utils import Preprocess

class GRU(nn.Module):
    transform = Preprocess(
        Preprocess.ToBezierCurves()
    )

    transform_plus = Preprocess(
        Preprocess.ToBezierCurves(),
        Preprocess.FeatureExtraction()
    )