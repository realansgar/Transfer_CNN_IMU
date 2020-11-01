import torch
from torch.utils.data import DataLoader
from datetime import datetime
import os
from tqdm import tqdm, trange
import CNN
from datasets import HARWindows
import metrics
from config import DEVICE, MODELS_BASEPATH, LOGS_BASEPATH, EVAL_FREQUENCY

# pylint: disable=no-member
class Trainer():
  def __init__(self, config, pretrained_state_dict={}, frozen_param_idxs=[]):
    self.config = config
    list(map(lambda item: setattr(self, *item), config.items()))

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
    self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.LEARNING_RATE, alpha=self.RMS_DECAY)

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

  def train(self, save=True):
    train_dataset = HARWindows(self.TRAIN_SET_FILEPATH)
    val_dataset = HARWindows(self.VAL_SET_FILEPATH)

    train_dataloader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)
    
    best_weights = None
    best_val_loss = float("inf")
    train_eval = []
    val_eval = []

    for epoch in trange(self.EPOCHS):
      train_data_pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
      train_eval_epoch = {}
      val_eval_epoch = {}
      for i, data in train_data_pbar:
        self.net.train()
        self.process_batch(data, noise=self.NOISE)
        train_data_pbar.set_description(f"epoch: {epoch + 1}/{self.EPOCHS}, training:")

        if i % EVAL_FREQUENCY == (EVAL_FREQUENCY - 1) or i == len(train_dataloader) - 1:
          train_data_pbar.set_description(f"epoch: {epoch + 1}/{self.EPOCHS}, validating:")
          train_eval_row = metrics.evaluate_net(self.net, self.criterion, data, self.NUM_CLASSES)
          val_eval_row = metrics.evaluate_net(self.net, self.criterion, next(iter(val_dataloader)), self.NUM_CLASSES)
          train_eval_epoch = {col: (train_eval_epoch[col] if col in train_eval_epoch else []) + [val] for (col, val) in train_eval_row.items()}
          val_eval_epoch = {col: (val_eval_epoch[col] if col in val_eval_epoch else []) + [val] for (col, val) in val_eval_row.items()}

          if val_eval_row["loss"] < best_val_loss:
            best_val_loss = val_eval_row["loss"]
            best_weights = self.net.state_dict()

      train_eval += [train_eval_epoch]      
      val_eval += [val_eval_epoch]

    if save:
      now = datetime.now()
      nowstr = now.strftime("%d.%m.%y_%H:%M:%S")
      best_net = self.Selected_CNN(self.config)
      best_net.load_state_dict(best_weights)
      final_val = metrics.evaluate_net(best_net, self.criterion, next(iter(val_dataloader)), self.NUM_CLASSES)
      filename = f"{self.NAME}_{nowstr}.{self.MODEL}.pt"
      os.makedirs(MODELS_BASEPATH, exist_ok=True)
      os.makedirs(LOGS_BASEPATH, exist_ok=True)
      torch.save(best_net, MODELS_BASEPATH + filename)
      torch.save({"train": train_eval, "val": val_eval, "config": self.config, "final_val": final_val}, LOGS_BASEPATH + filename)
    return self.net, final_val
