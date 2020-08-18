import torch
from torch.utils.data import Dataset
import numpy as np


class HARWindows(Dataset):
  '''
  classdocs
  '''
  def __init__(self, filepath):
    """
    :param filepath: filepath to .npz file with "data_x" and "data_y" arrays
    """
    np_dataset = np.load(filepath)
    self.data_x = torch.from_numpy(np_dataset["data_x"]).float()
    self.data_y = torch.from_numpy(np_dataset["data_y"]).long()
    if len(self.data_x) != len(self.data_y):
      raise ValueError("invalid dataset")
    

  def __len__(self):
    return len(self.data_x)

  def __getitem__(self, idx):
    """
    returns a window with its corresponding label

    :param idx: index of the window
    :returns tuple (data, label), where data has shape (1, rows, columns) and label has shape (1)
             the first axis corresponds to the depth of the data, which is 1 but gets larger it is forwarded through the network
    """
    return (self.data_x[idx:idx+1], self.data_y[idx])
