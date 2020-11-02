from torch import nn
from config import *


class CNN_IMU_Branch(nn.Module):
  def __init__(self, config, branch_size):
    super().__init__()
    list(map(lambda item: setattr(self, *item), config.items()))
    self.branch_size = branch_size

    conv_layers = [nn.Conv1d(in_channels=1, out_channels=self.NUM_KERNELS, kernel_size=(self.KERNEL_LENGTH, 1))]
    conv_layers += [nn.ReLU()]
    conv_layers += [nn.Conv1d(in_channels=self.NUM_KERNELS, out_channels=self.NUM_KERNELS, kernel_size=(self.KERNEL_LENGTH, 1))]
    conv_layers += [nn.ReLU()]
    conv_layers += [nn.MaxPool2d(kernel_size=(self.POOLING_LENGTH, 1), stride=(self.POOLING_STRIDE, 1))]
    conv_layers += [nn.Conv1d(in_channels=self.NUM_KERNELS, out_channels=self.NUM_KERNELS, kernel_size=(self.KERNEL_LENGTH, 1))]
    conv_layers += [nn.ReLU()]
    conv_layers += [nn.Conv1d(in_channels=self.NUM_KERNELS, out_channels=self.NUM_KERNELS, kernel_size=(self.KERNEL_LENGTH, 1))]
    conv_layers += [nn.ReLU()]
    conv_layers += [nn.MaxPool2d(kernel_size=(self.POOLING_LENGTH, 1), stride=(self.POOLING_STRIDE, 1))]
    self.convolutional_layers = nn.Sequential(*conv_layers)

    self.linear_feature_size = self.compute_linear_feature_size()
    fc_layers = [nn.Dropout(self.DROPOUT)]
    fc_layers += [nn.Linear(in_features=self.linear_feature_size, out_features=512)]
    fc_layers += [nn.ReLU()]
    self.fully_connected_layers = nn.Sequential(*fc_layers)

  def forward(self, x):
    x = self.convolutional_layers(x)
    x = x.view(-1, self.linear_feature_size)
    x = self.fully_connected_layers(x)
    return x

  def compute_linear_feature_size(self):
    x = torch.ones((1, 1, self.WINDOW_SIZE, self.branch_size)).float()
    with torch.no_grad():
      x = self.convolutional_layers(x)
    return torch.prod(torch.tensor(x.shape))


class Simple_CNN(nn.Module):
  def __init__(self, config):
    super().__init__()
    list(map(lambda item: setattr(self, *item), config.items()))

    self.cnn_imu_branch = CNN_IMU_Branch(config, self.NUM_SENSOR_CHANNELS)

    self.last_layer = nn.Linear(in_features=512, out_features=self.NUM_CLASSES)
  
  def forward(self, x):
    x = self.cnn_imu_branch(x)
    x = self.last_layer(x)
    return x
  

class CNN_IMU(nn.Module):
  def __init__(self, config):
    super().__init__()
    list(map(lambda item: setattr(self, *item), config.items()))
    
    self.imu_branches = nn.ModuleList()
    if isinstance(self.IMUS, list):
      num_imus = len(self.IMUS)
      for imu_size in self.IMUS:
        self.imu_branches.append(CNN_IMU_Branch(config, imu_size))
    else:
      num_imus = self.IMUS
      for _ in range(self.IMUS):
        self.imu_branches.append(CNN_IMU_Branch(config, self.NUM_SENSOR_CHANNELS // self.IMUS))

    comb_layers = [nn.Dropout(self.DROPOUT)]
    comb_layers += [nn.Linear(in_features=512*num_imus, out_features=512)]
    comb_layers += [nn.ReLU()] 
    comb_layers += [nn.Linear(in_features=512, out_features=self.NUM_CLASSES)]
    self.combining_layers = nn.Sequential(*comb_layers)
  
  def forward(self, x):
    if isinstance(self.IMUS, list):
      x_imus = torch.split(x, self.IMUS, dim=3)
    else:
      x_imus = torch.split(x, (self.NUM_SENSOR_CHANNELS // self.IMUS), dim=3)
    print(list(map(lambda x: x.shape, x_imus)))
    x = list(map(lambda imu_branch, x_imu: imu_branch(x_imu), self.imu_branches, x_imus))
    x = torch.cat(x, dim=1)
    x = self.combining_layers(x)
    return x

if __name__ == "__main__":
  from torch.utils.tensorboard import SummaryWriter
  wr = SummaryWriter()
  net = CNN_IMU(OPPORTUNITY_GESTURES)
  wr.add_graph(net, torch.ones(10,1,OPPORTUNITY_GESTURES["WINDOW_SIZE"], OPPORTUNITY_GESTURES["NUM_SENSOR_CHANNELS"]))
  wr.close()
