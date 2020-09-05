import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from tqdm import tqdm
import CNN
from datasets import HARWindows
import preprocessing
import metrics
from config import DEVICE, MODELS_BASEPATH, LOGS_BASEPATH, EVAL_FREQUENCY


class Trainer():
  def __init__(self, config_dict):
    self.config_dict = config_dict
    list(map(lambda item: setattr(self, *item), config_dict.items()))

  def process_batch(self, batch, optimize=True):
    data_batch, label_batch = batch
    data_batch = data_batch.to(DEVICE)
    label_batch = label_batch.to(DEVICE)
        
    # forward and backward pass
    if optimize:
      self.optimizer.zero_grad()
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
    
    Selected_CNN = getattr(CNN, self.MODEL)
    self.net = Selected_CNN(self.config_dict).to(DEVICE)

    self.criterion = torch.nn.CrossEntropyLoss()
    self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.LEARNING_RATE, alpha=self.RMS_DECAY)
    
    writer = SummaryWriter(LOGS_BASEPATH)
    best_weights = None
    best_val_loss = float("inf")
    train_eval = []
    val_eval = []

    for epoch in range(self.EPOCHS):
      train_data_pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
      train_eval_epoch = {}
      val_eval_epoch = {}
      for i, data in train_data_pbar:
        self.net.train()
        self.process_batch(data)
        train_data_pbar.set_description(f"epoch: {epoch}/{self.EPOCHS}, training:")

        if i % EVAL_FREQUENCY == (EVAL_FREQUENCY - 1):
          train_data_pbar.set_description(f"epoch: {epoch}/{self.EPOCHS}, validating:")
          train_eval_row = metrics.evaluate_net(self.net, self.criterion, data, self.NUM_CLASSES)
          val_eval_row = metrics.evaluate_net(self.net, self.criterion, next(iter(val_dataloader)), self.NUM_CLASSES)
          train_eval_epoch = {col: (train_eval_epoch[col] if col in train_eval_epoch else []) + [val] for (col, val) in train_eval_row.items()}
          val_eval_epoch = {col: (val_eval_epoch[col] if col in val_eval_epoch else []) + [val] for (col, val) in val_eval_row.items()}

          for col in train_eval_row:
            writer.add_scalars(col, {"train": train_eval_row[col], "validation": val_eval_row[col]}, i + epoch + len(train_dataloader))

          if val_eval_row["loss"] < best_val_loss:
            best_val_loss = val_eval_row["loss"]
            best_weights = self.net.state_dict()

      train_eval += [train_eval_epoch]      
      val_eval += [val_eval_epoch]

    writer.close()
    if save:
      now = datetime.now()
      nowstr = now.strftime("%d.%m.%y %H:%M:%S")
      best_net = Selected_CNN(self.config_dict)
      best_net.load_state_dict(best_weights)
      filename = f"{self.NAME}_{nowstr}.{self.MODEL}.pt"
      os.makedirs(MODELS_BASEPATH, exist_ok=True)
      os.makedirs(LOGS_BASEPATH, exist_ok=True)
      torch.save(best_net, MODELS_BASEPATH + filename)
      torch.save({"train": train_eval, "val": val_eval}, LOGS_BASEPATH + filename)
    return self.net
