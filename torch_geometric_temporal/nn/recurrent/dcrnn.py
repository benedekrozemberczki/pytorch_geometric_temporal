import math
import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.conv import MessagePassing

class AGCRN(object):

    def __init__(self):
        x = 1
