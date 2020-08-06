import torch
from torch import nn, optim
from CNN import SimpleCNN
import time
from config import *

class_weights = torch.ones(NUM_CLASSES)                                         # TODO weight classes by appearance in dataset
net = SimpleCNN()
net.is_train = True
optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, alpha=RMS_DECAY)
criterion = nn.CrossEntropyLoss(weight=class_weights)

starttime = time.time()
for i in range(10):                                                             # TODO use dataloader, create first training/test split
  optimizer.zero_grad()
  in_batch = torch.randn((BATCH_SIZE, 1, WINDOW_SIZE, NUM_SENSOR_CHANNELS))
  out_batch = net(in_batch)
  target_batch = torch.randint(0, NUM_CLASSES - 1, (BATCH_SIZE,))
  
  loss = criterion(out_batch, target_batch)
  loss.backward()
  optimizer.step()

endtime = time.time()
print(f"training time: {endtime - starttime}")