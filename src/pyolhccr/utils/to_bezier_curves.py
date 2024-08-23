from typing import List
import numpy as np
import torch

class ToBezierCurves:
    '''Completion time stamps of every points, dropout points if dropout is specified, 
    remove redundent points and normalize to 64*64, then transform to bezier curve representations
    Reference: https://arxiv.org/abs/1606.06539 (remove redundent points and normalize)
        https://arxiv.org/abs/1902.10525v2 (transform to bezier curve representations)
    '''
    def __init__(self, size=(64, 64), dropout=None):
        '''
        Args:
          droupout: Percentage of point random dropout.
        '''
        self.size = size
        self.dropout = dropout

    def __call__(self, x: List[np.ndarray]) -> torch.Tensor:

        # to torch
        for i, mat in enumerate(x):
            x[i] = torch.from_numpy(np.insert(mat, -1, i, axis=1))
        return x