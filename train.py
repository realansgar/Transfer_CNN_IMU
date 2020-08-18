import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from CNN import SimpleCNN
from datasets import HARWindows
import preprocessing
from config import *


def process_batch(batch, net, criterion, optimizer):
  data_batch, label_batch = data
  if USE_CUDA:
    data_batch = data_batch.cuda()
    label_batch = label_batch.cuda()
      
  # forward and backward pass
  optimizer.zero_grad()
  out_batch = net(data_batch)
  loss = criterion(out_batch, label_batch)
  loss.backward()
  optimizer.step()

  return loss

USE_CUDA = torch.cuda.is_available()

# set flags / seeds
torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)

if __name__ == '__main__':    
  train_dataset = HARWindows(PAMAP2_TRAIN_SET_FILEPATH)
  val_dataset = HARWindows(PAMAP2_VAL_SET_FILEPATH)
  test_dataset = HARWindows(PAMAP2_TEST_SET_FILEPATH)

  train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
  
  class_weights = torch.ones(NUM_CLASSES)                                         # TODO weight classes by appearance in dataset
  net = SimpleCNN()
  criterion = nn.CrossEntropyLoss(weight=class_weights)
  
  if USE_CUDA:
    net = net.cuda()
  
  optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, alpha=RMS_DECAY)
  
  # typically we use tensorboardX to keep track of experiments
  # writer = SummaryWriter(...)
  
  for epoch in range(PAMAP2_EPOCHS):
    net.train()

    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, data in pbar:
      start_time = time.time()
      loss = process_batch(data, net, criterion, optimizer)

      # udpate tensorboardX
      #writer.add_scalar(..., n_iter)

      process_time = time.time() - start_time
      pbar.set_description(f"Elapsed time: {process_time:.3f}s, loss: {loss.item()}, epoch: {epoch}/{PAMAP2_EPOCHS}:")
          
    # validation loss
    net.eval()
    with torch.no_grad():
      pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
      loss_acc = 0
      for i, data in pbar:
        start_time = time.time()
        loss = process_batch(data, net, criterion, optimizer)
        loss_acc += loss.item()
        process_time = time.time() - start_time
        pbar.set_description(f"Elapsed time: {process_time:.3f}s, loss: {loss.item()}, epoch: {epoch}/{PAMAP2_EPOCHS}:")
      print(f"Epoch {epoch}/{PAMAP2_EPOCHS} validation loss: total: {loss_acc}, mean: {loss_acc / len(val_dataloader)}")

  torch.save(net.state_dict(), MODELS_BASEPATH + "first_model.pt")