import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from itertools import chain
from tqdm import tqdm, trange
import CNN
from datasets import HARWindows
import metrics
import config as c
from config import DEVICE, EVAL_PERIOD

# pylint: disable=no-member
class Trainer():
  def __init__(self, config, pretrained_state_dict={}, frozen_param_idxs=[], conv_lr=None):
    self.config = config
    list(map(lambda item: setattr(self, *item), config.items()))

    if c.DETERMINISTIC:
      torch.manual_seed(42)

    self.Selected_CNN = getattr(CNN, self.MODEL)
    self.net = self.Selected_CNN(self.config).to(DEVICE)
    
    # Use orthogonal init for conv layers, default init otherwise
    for param in self.net.parameters():
      try:
        torch.nn.init.orthogonal_(param)
      except ValueError:
        pass

    self.net.load_state_dict(pretrained_state_dict, strict=False)

    for idx, param in enumerate(self.net.parameters()):
      if idx in frozen_param_idxs:
        param.requires_grad = False

    self.criterion = torch.nn.CrossEntropyLoss()

    if conv_lr is None:
      conv_lr = self.LEARNING_RATE

    all_params = set(self.net.parameters())
    if self.Selected_CNN == CNN.CNN_IMU:
      conv_params = set(chain(*[imu_branch.convolutional_layers.parameters() for imu_branch in self.net.imu_branches]))
    elif self.Selected_CNN == CNN.Simple_CNN:
      conv_params = set(self.net.cnn_imu_branch.convolutional_layers.parameters())
    not_conv_params = all_params - conv_params
    self.optimizer = torch.optim.RMSprop([
      {
        "params": list(conv_params),
        "lr": conv_lr
      },
      {
        "params": list(not_conv_params),
        "lr": self.LEARNING_RATE
      },
    ], alpha=self.RMS_DECAY)

    if self.NOISE:
      self.noise = torch.distributions.Normal(0, self.NOISE)

  def process_batch(self, batch, optimize=True, noise=True):
    data_batch, label_batch = batch
    data_batch = data_batch.to(DEVICE)
    label_batch = label_batch.to(DEVICE)

    # forward and backward pass
    if optimize:
      self.optimizer.zero_grad()
    if noise:
      added_noise = self.noise.sample(data_batch.shape)
      added_noise = added_noise.to(DEVICE)
      data_batch += added_noise
    out_batch = self.net(data_batch)
    loss = self.criterion(out_batch, label_batch)
    if optimize:
      loss.backward()
      self.optimizer.step()

    return loss

  def train(self):
    train_dataset = HARWindows(self.TRAIN_SET_FILEPATH)
    val_dataset = HARWindows(self.VAL_SET_FILEPATH)

    train_dataloader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))
    
    best_weights_acc = None
    best_weights_wf1 = None
    best_val_acc = float("-inf")
    best_val_wf1 = float("-inf")
    best_epoch_acc = -1
    best_iteration_acc = -1
    best_epoch_wf1 = -1
    best_iteration_wf1 = -1
    train_eval = []
    val_eval = []

    for epoch in trange(self.EPOCHS, desc="epochs"):
      train_data_pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
      train_eval_epoch = {}
      val_eval_epoch = {}
      for i, data in train_data_pbar:
        self.net.train()
        self.process_batch(data, noise=self.NOISE)
        train_data_pbar.set_description("training  ")

        if i % EVAL_PERIOD == (EVAL_PERIOD - 1) or i == len(train_dataloader) - 1:
          train_data_pbar.set_description("validating")
          train_eval_row = metrics.evaluate_net(self.net, self.criterion, data, self.NUM_CLASSES)
          val_eval_row = metrics.evaluate_net(self.net, self.criterion, next(iter(val_dataloader)), self.NUM_CLASSES)
          train_eval_epoch = {col: (train_eval_epoch[col] if col in train_eval_epoch else []) + [val] for (col, val) in train_eval_row.items()}
          val_eval_epoch = {col: (val_eval_epoch[col] if col in val_eval_epoch else []) + [val] for (col, val) in val_eval_row.items()}

          if val_eval_row["micro_accuracy"] > best_val_acc:
            best_val_acc = val_eval_row["micro_accuracy"]
            best_weights_acc = deepcopy(self.net.state_dict())
            best_epoch_acc = epoch
            best_iteration_acc = i
          if val_eval_row["weighted_f1"] > best_val_wf1:
            best_val_wf1 = val_eval_row["weighted_f1"]
            best_weights_wf1 = deepcopy(self.net.state_dict())
            best_epoch_wf1 = epoch
            best_iteration_wf1 = i

      train_eval += [train_eval_epoch]      
      val_eval += [val_eval_epoch]

    best_net_acc = self.Selected_CNN(self.config)
    best_net_acc.load_state_dict(best_weights_acc)
    best_net_wf1 = self.Selected_CNN(self.config)
    best_net_wf1.load_state_dict(best_weights_wf1)
    best_val_acc = metrics.evaluate_net(best_net_acc, self.criterion, next(iter(val_dataloader)), self.NUM_CLASSES)
    best_val_wf1 = metrics.evaluate_net(best_net_wf1, self.criterion, next(iter(val_dataloader)), self.NUM_CLASSES)
    eval_dict_acc = {"net": best_net_acc, "train": train_eval, "val": val_eval, "config": self.config, "best_val": best_val_acc, "best_epoch": best_epoch_acc, "best_iteration": best_iteration_acc}
    eval_dict_wf1 = {"net": best_net_wf1, "train": train_eval, "val": val_eval, "config": self.config, "best_val": best_val_wf1, "best_epoch": best_epoch_wf1, "best_iteration": best_iteration_wf1}
    return eval_dict_acc, eval_dict_wf1
