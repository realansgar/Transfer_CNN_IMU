import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from train import Trainer
from config import PAMAP2, LOGS_BASEPATH, DEVICE, EVAL_FREQUENCY

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
    ax.plot(range(0, len(train_eval[key]) * EVAL_FREQUENCY, EVAL_FREQUENCY), train_eval[key], label="train")
    ax.plot(range(0, len(val_eval[key]) * EVAL_FREQUENCY, EVAL_FREQUENCY), val_eval[key], label="validation")
    ax.set_xlabel("iterations")
    ax.set_ylabel(key)
    ax.set_title(key)
    ax.legend()
  title = os.path.basename(filename)
  fig.set_size_inches(23.3, 16.5)
  fig.tight_layout()
  fig.savefig(os.path.splitext(filename)[0] + ".pdf", orientation="landscape", bbox_inches='tight')


def pamap2_hyperparameters():
  pamap2_cnn_imu_2 = PAMAP2.copy()
  pamap2_trainer = Trainer(pamap2_cnn_imu_2)
  pamap2_trainer.train()

  pamap2_simple_cnn = pamap2_cnn_imu_2.copy()
  pamap2_simple_cnn["NAME"] = "PAMAP2 - SimpleCNN"
  pamap2_simple_cnn["MODEL"] = "SimpleCNN"
  pamap2_trainer = Trainer(pamap2_cnn_imu_2)
  pamap2_trainer.train()
