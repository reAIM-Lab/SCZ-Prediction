import os
import numpy as np
import pandas as pd
from collections import Counter
import torch
from sklearn.metrics import *
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import sys
import itertools

class Weighted_BCELoss(nn.Module):
    def __init__(self, weights, eps=1e-6):
        super(Weighted_BCELoss, self).__init__()
        self.weights = weights
        self.eps = eps

    def forward(self, output, target, smooth=1):
        output = torch.clamp(output, self.eps, 1 - self.eps)
        loss = self.weights[1] * (target * torch.log(output)) + self.weights[0] * ((1 - target) * torch.log(1 - output))
        return -torch.mean(loss)
