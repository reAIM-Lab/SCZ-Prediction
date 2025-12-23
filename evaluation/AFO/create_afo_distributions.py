import random
import torch
import os
import numpy as np
import pandas as pd
import pickle
import numpy as np
from tqdm import tnrange, tqdm_notebook
import torch.nn as nn
import time

seed_value = 35
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)

# set up dataset and paths

def read_write_means(train_loader_path, save_directory):
    train_loader = torch.load(train_loader_path, weights_only=False)
    print('done loading train loader')
    data_pts = torch.cat([x[1] for x in train_loader])
    data_distribution = data_pts.permute(0, 2, 1) # pt, feat, time
    print(data_distribution.shape)
    print('done making data dist')
    torch.save(data_distribution, f'{save_directory}/data_points_distribution.pt')

train_loader_path = f'{int_path}/{dataset_prefix}train_loader.pth'
read_write_means(train_loader_path, save_directory)
