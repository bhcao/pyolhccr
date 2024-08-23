from typing import List
import numpy as np
import torch

class Rasterization:
    '''
    Reference: https://ieeexplore.ieee.org/document/8949703
    '''
    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, x: List[np.ndarray]) -> torch.Tensor:
        pass