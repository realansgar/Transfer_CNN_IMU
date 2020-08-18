from torch import nn
from torch.nn import functional as F
import numpy as np
from config import *


def compute_linear_feature_size(layers):
  feature_size = [1, WINDOW_SIZE, NUM_SENSOR_CHANNELS]
  for layer in layers:
    if isinstance(layer, nn.Conv1d):
      feature_size[0] = layer.out_channels
      feature_size[1] -= (layer.kernel_size[0] - 1)
    if isinstance(layer, nn.MaxPool2d):
      feature_size[1] //= layer.kernel_size[0]
  return np.prod(feature_size)


class SimpleCNN(nn.Module):
  def __init__(self):
    super().__init__()

    conv_layers = [nn.Conv1d(in_channels=1, out_channels=NUM_KERNELS, kernel_size=(KERNEL_LENGTH, 1))]
    conv_layers += [nn.ReLU()]

    # should this be here? It isn't mentioned in the paper
    conv_layers += [nn.LocalResponseNorm(size=5)]

    conv_layers += [nn.Conv1d(in_channels=NUM_KERNELS, out_channels=NUM_KERNELS, kernel_size=(KERNEL_LENGTH, 1))]
    conv_layers += [nn.ReLU()]
    conv_layers += [nn.MaxPool2d(kernel_size=(POOLING_LENGTH, 1))]
    conv_layers += [nn.Conv1d(in_channels=NUM_KERNELS, out_channels=NUM_KERNELS, kernel_size=(KERNEL_LENGTH, 1))]
    conv_layers += [nn.ReLU()]
    conv_layers += [nn.Conv1d(in_channels=NUM_KERNELS, out_channels=NUM_KERNELS, kernel_size=(KERNEL_LENGTH, 1))]
    conv_layers += [nn.ReLU()]
    conv_layers += [nn.MaxPool2d(kernel_size=(POOLING_LENGTH, 1))]
    self.conv_net = nn.Sequential(*conv_layers)

    self.linear_feature_size = compute_linear_feature_size(conv_layers)
    fc_layers = [nn.Linear(in_features=self.linear_feature_size, out_features=512)]
    fc_layers += [nn.ReLU()]
    fc_layers += [nn.Linear(in_features=512, out_features=NUM_CLASSES)]
    self.fc_net = nn.Sequential(*fc_layers)

  def forward(self, x):
    x = self.conv_net(x)
    x = x.view(-1, self.linear_feature_size)
    x = self.fc_net(x)
    if not self.training:
      x = F.softmax(x, dim=1)
    return x
