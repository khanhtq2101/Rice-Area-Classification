import pandas as pd
import numpy as np
import torch
import os

land_type = ('Crop', 'Forest', 'Rice', 'Urban', 'Water')

class PointTimeSeries(torch.utils.data.Dataset):
  def __init__(self, root, polarlization = 'both', is_train= True):
    self.root = root
    self.polarlization = polarlization

    train_test_idx = torch.load(os.path.join(root, 'data/train_test_idx.pt'))
    if is_train:
      point_idx = train_test_idx['train_idx']
    else:
      point_idx = train_test_idx['test_idx']

    self.vv_frame_data = pd.read_csv(os.path.join(root, 'data/VV_data.csv')).iloc[point_idx, :]
    self.vh_frame_data = pd.read_csv(os.path.join(root, 'data/VH_data.csv')).iloc[point_idx, :]
  
  def __len__(self):
    return len(self.vv_frame_data)

  def __getitem__(self, idx):
    vv_series = torch.tensor(self.vv_frame_data.iloc[idx, 2: -1], dtype= torch.float32)[None, :]
    vh_series = torch.tensor(self.vh_frame_data.iloc[idx, 2: -1], dtype= torch.float32)[None, :]
    
    if self.polarlization == 'vv':
      series = vv_series
    elif self.polarlization == 'vh':
      series = vh_series
    else:
      series = torch.cat((vv_series, vh_series), dim = 0)
    
    series = torch.nn.functional.normalize(series, dim= 1)
    label = land_type.index(self.vv_frame_data.iloc[idx, -1])

    return series, label



def generate_train_test_idx(train_ratio, root):
  vv_data = pd.read_csv(os.path.join(root, 'data/VV_data.csv'))
  train_idx = np.array([], dtype= int)
  test_idx = np.array([], dtype= int)

  for i in range(len(land_type)):
    land_idx = vv_data[vv_data.Type == land_type[i]].index
    n_train = int(len(land_idx) * train_ratio[i])

    train_idx_i = np.random.choice(land_idx, n_train, replace= False)
    test_idx_i = np.setxor1d(land_idx, train_idx_i)

    train_idx = np.concatenate((train_idx, train_idx_i))
    test_idx = np.concatenate((test_idx, test_idx_i))

    train_test_idx = {'train_idx': train_idx,
                      'test_idx': test_idx}
    torch.save(train_test_idx, os.path.join(root, 'data/train_test_idx.pt'))

  return train_idx, test_idx