from torch import nn
from torch.nn import functional as F
import numpy as np
from config import *
from torch.utils.tensorboard import SummaryWriter


class CNN_IMU_Branch(nn.Module):
  def __init__(self, config_dict, is_simple=False):
    super().__init__()
    list(map(lambda item: setattr(self, *item), config_dict.items()))
    self.is_simple = is_simple

    conv_layers = [nn.Conv1d(in_channels=1, out_channels=self.NUM_KERNELS, kernel_size=(self.KERNEL_LENGTH, 1))]
    conv_layers += [nn.ReLU()]
    conv_layers += [nn.Conv1d(in_channels=self.NUM_KERNELS, out_channels=self.NUM_KERNELS, kernel_size=(self.KERNEL_LENGTH, 1))]
    conv_layers += [nn.ReLU()]
    conv_layers += [nn.MaxPool2d(kernel_size=(self.POOLING_LENGTH, 1))]
    conv_layers += [nn.Conv1d(in_channels=self.NUM_KERNELS, out_channels=self.NUM_KERNELS, kernel_size=(self.KERNEL_LENGTH, 1))]
    conv_layers += [nn.ReLU()]
    conv_layers += [nn.Conv1d(in_channels=self.NUM_KERNELS, out_channels=self.NUM_KERNELS, kernel_size=(self.KERNEL_LENGTH, 1))]
    conv_layers += [nn.ReLU()]
    conv_layers += [nn.MaxPool2d(kernel_size=(self.POOLING_LENGTH, 1))]
    self.convolutional_layers = nn.Sequential(*conv_layers)

    self.linear_feature_size = self.compute_linear_feature_size(conv_layers)
    fc_layers = [nn.Linear(in_features=self.linear_feature_size, out_features=512)]
    fc_layers += [nn.ReLU()]
    fc_layers += [nn.Dropout(self.DROPOUT)]
    self.fully_connected_layers = nn.Sequential(*fc_layers)

  def forward(self, x):
    x = self.convolutional_layers(x)
    x = x.view(-1, self.linear_feature_size)
    x = self.fully_connected_layers(x)
    return x

  def compute_linear_feature_size(self, layers):
    x = (self.NUM_IMUS if not self.is_simple else 1)
    feature_size = [1, self.WINDOW_SIZE, (self.NUM_SENSOR_CHANNELS // x)]
    for layer in layers:
      if isinstance(layer, nn.Conv1d):
        feature_size[0] = layer.out_channels
        feature_size[1] -= (layer.kernel_size[0] - 1)
      if isinstance(layer, nn.MaxPool2d):
        feature_size[1] //= layer.kernel_size[0]
    return np.prod(feature_size)


class SimpleCNN(nn.Module):
  def __init__(self, config_dict):
    super().__init__()
    list(map(lambda item: setattr(self, *item), config_dict.items()))

    self.cnn_imu_branch = CNN_IMU_Branch(config_dict, is_simple=True)

    self.last_layer = nn.Linear(in_features=512, out_features=self.NUM_CLASSES)
  
  def forward(self, x):
    x = self.cnn_imu_branch(x)
    x = self.last_layer(x)
    if not self.training:
      x = F.softmax(x, dim=1)
    return x
  

class CNN_IMU(nn.Module):
  def __init__(self, config_dict):
    super().__init__()
    list(map(lambda item: setattr(self, *item), config_dict.items()))
    
    self.imu_branches = nn.ModuleList()
    for i in range(self.NUM_IMUS):
      self.imu_branches.append(CNN_IMU_Branch(config_dict))
    
    comb_layers = [nn.Linear(in_features=512*self.NUM_IMUS, out_features=512)]
    comb_layers += [nn.ReLU()] 
    comb_layers += [nn.Dropout(self.DROPOUT)]
    comb_layers += [nn.Linear(in_features=512, out_features=self.NUM_CLASSES)]
    self.combining_layers = nn.Sequential(*comb_layers)
  
  def forward(self, x):
    x_imus = torch.split(x, (self.NUM_SENSOR_CHANNELS // self.NUM_IMUS), dim=3)
    x = list(map(lambda imu_branch, x_imu: imu_branch(x_imu), self.imu_branches, x_imus))
    x = torch.cat(x, dim=1)
    x = self.combining_layers(x)
    if not self.training:
      x = F.softmax(x, dim=1)
    return x

if __name__ == "__main__":
  from torch.utils.tensorboard import SummaryWriter
  net = CNN_IMU(PAMAP2)
  wr = SummaryWriter(log_dir="runs")
  wr.add_graph(net, torch.ones((10,1,100,27)))
  wr.close()
