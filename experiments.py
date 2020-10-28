import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from itertools import chain
from train import Trainer
import CNN
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

def filter_state_dict(state_dict):
  return {k: v for k, v in state_dict.items() if "convolutional_layers" in k}

def determine_frozen_param_idxs(state_dict, layer_num):
  state_dict = filter_state_dict(state_dict)
  keys = list(state_dict.keys())
  conv_layer_idx_pattern = re.compile(r".*?(\d)\D*$") # matches the last digit to determine the conv_layer idx
  idxs = sorted(list({conv_layer_idx_pattern.findall(k)[0] for k in keys}))
  conv_layers = {k: [] for k in idxs}
  for k in keys:
    idx = conv_layer_idx_pattern.findall(k)[0]
    conv_layers[idx].append(k)
  freeze_layer_idxs = list(conv_layers.keys())[:layer_num]
  freeze_keys = list(chain(*[conv_layers[idx] for idx in freeze_layer_idxs]))
  freeze_idx = [keys.index(freeze_key) for freeze_key in freeze_keys]
  return sorted(freeze_idx)


def pamap2_hyperparameters():
  pamap2_cnn_imu_2 = PAMAP2.copy()
  pamap2_trainer = Trainer(pamap2_cnn_imu_2)
  pamap2_trainer.train()

  pamap2_simple_cnn = pamap2_cnn_imu_2.copy()
  pamap2_simple_cnn["NAME"] = "PAMAP2 - SimpleCNN"
  pamap2_simple_cnn["MODEL"] = "SimpleCNN"
  pamap2_trainer = Trainer(pamap2_cnn_imu_2)
  pamap2_trainer.train()
