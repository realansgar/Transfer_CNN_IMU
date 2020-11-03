import re
from itertools import chain
import os
from argparse import ArgumentParser
import torch
from train import Trainer
import config
from config import LOGS_BASEPATH

def save_best_result(results, name, key):
  best_loss = float("inf")
  best_result = None
  for result in results:
    if result["best_val"]["loss"] < best_loss:
      best_result = result
  eval_dict = best_result
  filename = f"{name}-{eval_dict['config'][key]}_best.pt"
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


def base_hyperparameter(dataset, key, values):
  config_dict = getattr(config, dataset).copy()
  for model in ["Simple_CNN", "CNN_IMU"]:
    results = []
    name = f"{dataset}-{model}-{key}"
    for value in values:
      config_dict["NAME"] = f"{name}-{value}"
      config_dict["MODEL"] = model
      config_dict[key] = value
      print(f"-----{config_dict['NAME']}-----")
      trainer = Trainer(config_dict)
      eval_dict = trainer.train()
      results.append(eval_dict)
      print(eval_dict["best_val"], f"epoch: {eval_dict['best_epoch']}, iteration: {eval_dict['best_iteration']}\n")
    save_best_result(results, name, key)

def base_hyperparameter_order_picking(dataset, key, values):
  subject_re = re.compile(r"subject\d\d\d")
  config_dict = getattr(config, dataset).copy()
  for train_filepath, val_filepath in getattr(config, f"{dataset}_TRAIN_VAL_SET_FILEPATHS"):
    subject = subject_re.findall(val_filepath)[0]
    for model in ["Simple_CNN", "CNN_IMU"]:
      results = []
      name = f"{dataset}-{subject}-{model}-{key}"
      for value in values:
        config_dict["TRAIN_SET_FILEPATH"] = train_filepath
        config_dict["VAL_SET_FILEPATH"] = val_filepath
        config_dict["NAME"] = f"{name}-{value}"
        config_dict["MODEL"] = model
        config_dict[key] = value
        trainer = Trainer(config_dict)
        eval_dict = trainer.train()
        results.append(eval_dict)
        print(eval_dict["best_val"], f"epoch: {eval_dict['best_epoch']}, iteration: {eval_dict['best_iteration']}\n")
      save_best_result(results, name, key)


def pamap2_epochs():
  base_hyperparameter("PAMAP2", "EPOCHS", [20,12,8])

def pamap2_learning_rate():
  base_hyperparameter("PAMAP2", "LEARNING_RATE", [10**-3, 10**-4, 10**-5])

def opportunity_locomotion_epochs():
  base_hyperparameter("OPPORTUNITY_LOCOMOTION", "EPOCHS", [20,12,8])

def opportunity_locomotion_learning_rate():
  base_hyperparameter("OPPORTUNITY_LOCOMOTION", "LEARNING_RATE", [10**-3, 10**-4, 10**-5])

def opportunity_gestures_epochs():
  base_hyperparameter("OPPORTUNITY_GESTURES", "EPOCHS", [20,12,8])

def opportunity_gestures_learning_rate():
  base_hyperparameter("OPPORTUNITY_LOCOMOTION", "LEARNING_RATE", [10**-3, 10**-4, 10**-5])

def order_picking_a_epochs():
  base_hyperparameter_order_picking("ORDER_PICKING_A", "EPOCHS", [25,20,15])

def order_picking_a_learning_rate():
  base_hyperparameter_order_picking("ORDER_PICKING_A", "LEARNING_RATE", [10**-4, 10**-5, 10**-6, 10**-7])

def order_picking_b_epochs():
  base_hyperparameter_order_picking("ORDER_PICKING_B", "EPOCHS", [25,20,15])

def order_picking_b_learning_rate():
  base_hyperparameter_order_picking("ORDER_PICKING_B", "LEARNING_RATE", [10**-4, 10**-5, 10**-6, 10**-7])

def all_hyperparameters():
  pamap2_epochs()
  pamap2_learning_rate()
  opportunity_locomotion_epochs()
  opportunity_locomotion_learning_rate()
  opportunity_gestures_epochs()
  opportunity_gestures_learning_rate()
  order_picking_a_epochs()
  order_picking_a_learning_rate()
  order_picking_b_epochs()
  order_picking_b_learning_rate()

if __name__ == "__main__":
  parser = ArgumentParser(description="Start predefined experiments")
  parser.add_argument("experiment", help="the experiment to start")
  args = parser.parse_args()
  globals()[args.experiment]()