import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from CNN import SimpleCNN
from datasets import HARWindows
import preprocessing
from config import DEVICE

class Trainer():
  def __init__(self, config_dict):
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

  def train(self):
    train_dataset = HARWindows(self.TRAIN_SET_FILEPATH)
    val_dataset = HARWindows(self.VAL_SET_FILEPATH)
    test_dataset = HARWindows(self.TEST_SET_FILEPATH)

    train_dataloader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    
    writer = SummaryWriter(self.LOGS_BASEPATH)

    self.net = SimpleCNN().to(DEVICE)
    class_weights = torch.ones(self.NUM_CLASSES, device=DEVICE)                                  # TODO weight classes by appearance in dataset

    self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.LEARNING_RATE, alpha=self.RMS_DECAY)
    
    # typically we use tensorboardX to keep track of experiments
    # writer = SummaryWriter(...)
    
    for epoch in range(self.EPOCHS):
      self.net.train()

      pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
      for i, data in pbar:
        start_time = time.time()
        loss = self.process_batch(data)

        # udpate tensorboardX
        #writer.add_scalar(..., n_iter)

        process_time = time.time() - start_time
        pbar.set_description(f"loss: {loss.item()}, epoch: {epoch}/{self.EPOCHS}:")
            
      # validation loss
      self.net.eval()
      with torch.no_grad():
        pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        loss_acc = 0
        for i, data in pbar:
          start_time = time.time()
          loss = self.process_batch(data, optimize=False)
          loss_acc += loss.item()
          process_time = time.time() - start_time
          pbar.set_description(f"loss: {loss.item()}, epoch: {epoch}/{self.EPOCHS}:")
        print(f"Epoch {epoch}/{self.EPOCHS} validation loss: total: {loss_acc}, mean: {loss_acc / len(val_dataloader)}")

  def evaluate(self, dataset):
    pass  
