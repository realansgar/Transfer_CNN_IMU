import re
from itertools import chain
import os
from argparse import ArgumentParser
import torch
from train import Trainer
from config import *

def save_best_result(results, name):
  best_loss = float("inf")
  best_result = None
  for result in results:
    if result["best_val"]["loss"] < best_loss:
      best_result = result
  eval_dict = best_result
  filename = f"{name}_best.{eval_dict['config']['MODEL']}.pt"
  os.makedirs(LOGS_BASEPATH, exist_ok=True)
  torch.save(eval_dict, LOGS_BASEPATH + filename)


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


def pamap2_epochs():
  pamap2_simple_cnn_results = []
  for epochs in [20,12,8]:
    pamap2_simple_cnn = PAMAP2.copy()
    pamap2_simple_cnn["NAME"] = f"PAMAP2-Simple_CNN-{epochs}ep"
    pamap2_simple_cnn["MODEL"] = "Simple_CNN"
    pamap2_simple_cnn["EPOCHS"] = epochs
    pamap2_trainer = Trainer(pamap2_simple_cnn)
    eval_dict = pamap2_trainer.train()
    pamap2_simple_cnn_results.append(eval_dict)
    print(eval_dict["best_val"])
  save_best_result(pamap2_simple_cnn_results, "PAMAP2-Simple_CNN-ep")

  pamap2_cnn_imu_results = []
  for epochs in [20,12,8]:
    pamap2_cnn_imu = PAMAP2.copy()
    pamap2_cnn_imu["NAME"] = f"PAMAP2-CNN_IMU-{epochs}ep"
    pamap2_cnn_imu["MODEL"] = "CNN_IMU"
    pamap2_cnn_imu["EPOCHS"] = epochs
    pamap2_trainer = Trainer(pamap2_cnn_imu)
    eval_dict = pamap2_trainer.train()
    pamap2_cnn_imu_results.append(eval_dict)
    print(eval_dict["best_val"])
  save_best_result(pamap2_cnn_imu_results, "PAMAP2-CNN_IMU-ep")

if __name__ == "__main__":
  parser = ArgumentParser(description="Start predefined experiments")
  parser.add_argument("experiment", help="the experiment to start")
  args = parser.parse_args()
  globals()[args.experiment]()