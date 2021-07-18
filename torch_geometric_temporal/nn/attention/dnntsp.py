import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.utils.to_dense_adj import to_dense_adj
import torch.nn.functional as F

class DNNTSP(nn.Module):

     def __init__(self):
         self.num = 32
