from typing import List
import numpy as np
import torch

from pyolhccr.utils import feature_extraction, rasterization, to_bezier_curves

class Preprocess:
    '''Call functions and return their results in list.'''
    def __init__(self, *functions):
        self.functions = functions
    
    # alias for uniform use
    ToBezierCurves = to_bezier_curves.ToBezierCurves
    Rasterization = rasterization.Rasterization
    FeatureExtraction = feature_extraction.FeatureExtraction

    def __call__(self, x: List[np.ndarray]) -> List[torch.Tensor]:
        output = []
        for function in self.functions:
            output.append(function(x))
        return output