import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from train import Trainer
from config import PAMAP2, LOGS_BASEPATH, DEVICE

def plot(filename):
  eval_dict = torch.load(filename, map_location=DEVICE)
  train_eval = eval_dict["train"]
  val_eval = eval_dict["val"]
  # [{"a": [1,2], "b": [3,4]}, {"a": [5,6], "b": [7,8]}] -> {"a": [1,2,5,6], "b": [3,4,7,8]} losing information about epochs
  train_eval = {key: [item for sublist in [d[key] for d in train_eval] for item in sublist] for key in train_eval[0]}
  val_eval = {key: [item for sublist in [d[key] for d in val_eval] for item in sublist] for key in val_eval[0]}
  fig, axs = plt.subplots(len(train_eval) // 2, 2)
  axs = axs.flatten()
  for (i, key) in enumerate(train_eval):
    ax = axs[i]
    ax.plot(range(len(train_eval[key])), train_eval[key], label="train")
    ax.plot(range(len(val_eval[key])), val_eval[key], label="validation")
    ax.set_xlabel("iterations")
    ax.set_ylabel(key)
    ax.set_title(key)
    ax.legend()
  title = os.path.basename(filename)
  #fig.set_title(title)
  fig.savefig(os.path.splitext(filename)[0] + ".pdf", bbox_inches='tight')


def pamap2_hyperparameters():
  pamap2_trainer = Trainer(PAMAP2)
  pamap2_trainer.train()
