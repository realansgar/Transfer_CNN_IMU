import re
from itertools import chain
import os
from argparse import ArgumentParser
import torch
from train import Trainer
import config
from config import LOGS_BASEPATH, DEVICE

SAVEALL = False
subject_re = re.compile(r"subject\d\d\d")

def save_best_result(results, name, key):
  os.makedirs(LOGS_BASEPATH, exist_ok=True)
  best_acc = float("-inf")
  best_wf1 = float("-inf")
  best_result_acc = None
  best_result_wf1 = None
  for result in results:
    eval_dict_acc, eval_dict_wf1 = result
    if SAVEALL:
      filename = f"{name}-{eval_dict_acc['config'][key]}_acc_{eval_dict_acc['best_val']['micro_accuracy']:.4f}.pt"
      torch.save(eval_dict_acc, LOGS_BASEPATH + filename)
      filename = f"{name}-{eval_dict_wf1['config'][key]}_wf1_{eval_dict_wf1['best_val']['weighted_f1']:.4f}.pt"
      torch.save(eval_dict_wf1, LOGS_BASEPATH + filename)
    if eval_dict_acc["best_val"]["micro_accuracy"] > best_acc:
      best_acc = eval_dict_acc["best_val"]["micro_accuracy"]
      best_result_acc = eval_dict_acc
    if eval_dict_wf1["best_val"]["weighted_f1"] > best_wf1:
      best_wf1 = eval_dict_wf1["best_val"]["weighted_f1"]
      best_result_wf1 = eval_dict_wf1
  filename = f"{name}-{best_result_acc['config'][key]}_best_acc_{best_result_acc['best_val']['micro_accuracy']:.4f}_epoch_{best_result_acc['best_epoch']}_iteration_{best_result_acc['best_iteration']}.pt"
  torch.save(best_result_acc, LOGS_BASEPATH + filename)
  filename = f"{name}-{best_result_wf1['config'][key]}_best_wf1_{best_result_wf1['best_val']['weighted_f1']:.4f}_epoch_{best_result_wf1['best_epoch']}_iteration_{best_result_wf1['best_iteration']}.pt"
  torch.save(best_result_wf1, LOGS_BASEPATH + filename)


def filter_state_dict(state_dict):
  return {k: v for k, v in state_dict.items() if "convolutional_layers" in k}

def determine_param_idx(state_dict, layer_num, keys=False):
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
  if keys:
    return freeze_keys
  freeze_idx = [keys.index(freeze_key) for freeze_key in freeze_keys]
  return sorted(freeze_idx)


def base_hyperparameter(dataset, key, values):
  for model in ["Simple_CNN", "CNN_IMU"]:
    results = []
    name = f"{dataset}-{model}-{key}"
    for value in values:
      config_dict = getattr(config, dataset).copy()
      config_dict["NAME"] = f"{name}-{value}"
      config_dict["MODEL"] = model
      config_dict[key] = value
      print(f"-----{config_dict['NAME']}-----")
      trainer = Trainer(config_dict)
      eval_dict = trainer.train()
      results.append(eval_dict)
      eval_dict_acc, eval_dict_wf1 = eval_dict
      print("ACC: ", eval_dict_acc["best_val"], f"epoch: {eval_dict_acc['best_epoch']}, iteration: {eval_dict_acc['best_iteration']}\n")
      print("WF1: ", eval_dict_wf1["best_val"], f"epoch: {eval_dict_wf1['best_epoch']}, iteration: {eval_dict_wf1['best_iteration']}\n")
    save_best_result(results, name, key)

def base_hyperparameter_order_picking(dataset, key, values):
  for train_filepath, val_filepath in getattr(config, f"{dataset}_TRAIN_VAL_SET_FILEPATHS"):
    subject = subject_re.findall(val_filepath)[0]
    for model in ["Simple_CNN", "CNN_IMU"]:
      results = []
      name = f"{dataset}-{subject}-{model}-{key}"
      for value in values:
        config_dict = getattr(config, dataset).copy()
        config_dict["TRAIN_SET_FILEPATH"] = train_filepath
        config_dict["VAL_SET_FILEPATH"] = val_filepath
        config_dict["NAME"] = f"{name}-{value}"
        config_dict["MODEL"] = model
        config_dict[key] = value
        trainer = Trainer(config_dict)
        eval_dict = trainer.train()
        results.append(eval_dict)
        eval_dict_acc, eval_dict_wf1 = eval_dict
        print("ACC: ", eval_dict_acc["best_val"], f"epoch: {eval_dict_acc['best_epoch']}, iteration: {eval_dict_acc['best_iteration']}\n")
        print("WF1: ", eval_dict_wf1["best_val"], f"epoch: {eval_dict_wf1['best_epoch']}, iteration: {eval_dict_wf1['best_iteration']}\n")
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
  base_hyperparameter("OPPORTUNITY_GESTURES", "LEARNING_RATE", [10**-3, 10**-4, 10**-5])

def order_picking_a_epochs():
  base_hyperparameter_order_picking("ORDER_PICKING_A", "EPOCHS", [25,20,15])

def order_picking_a_learning_rate():
  base_hyperparameter_order_picking("ORDER_PICKING_A", "LEARNING_RATE", [10**-4, 10**-5, 10**-6, 10**-7])

def order_picking_b_epochs():
  base_hyperparameter_order_picking("ORDER_PICKING_B", "EPOCHS", [25,20,15])

def order_picking_b_learning_rate():
  base_hyperparameter_order_picking("ORDER_PICKING_B", "LEARNING_RATE", [10**-4, 10**-5, 10**-6, 10**-7])

def all_hyperparameters():
  pamap2_learning_rate()
  opportunity_locomotion_learning_rate()
  opportunity_gestures_learning_rate()
  order_picking_a_learning_rate()
  order_picking_b_learning_rate()


def base_transfer(source_dataset, target_dataset, model, freeze=0, mapping=None, layer_num=None):
  filepath = getattr(config, f"{source_dataset}_BEST_{model}")
  eval_dict = torch.load(filepath, map_location=DEVICE)
  state_dict = eval_dict["net"].state_dict()
  state_dict = filter_state_dict(state_dict)
  source_config = getattr(config, source_dataset)
  target_config = getattr(config, target_dataset)
  if mapping is not None:
    imu_branch_pattern = re.compile(r"imu_branches\.(\d+)\..*")
    idx_mapping = {source_config["BRANCHES"].index(k): [target_config["BRANCHES"].index(t) for t in v] for k, v in mapping.items()}
    new_state_dict = {}
    for source_idx, target_idxs in idx_mapping.items():
      keys = []
      for k in state_dict:
        if imu_branch_pattern.findall(k)[0] == str(source_idx):
          keys.append(k)
      imu_branch_state_dict = {k: v for k, v in state_dict.items() if k in keys}
      for target_idx in target_idxs:
        new_state_dict.update({k.replace(str(source_idx), str(target_idx), 1): v for k, v in imu_branch_state_dict.items()})
    state_dict = new_state_dict
  
  freeze_idx = determine_param_idx(state_dict, freeze)
  if layer_num is not None:
    keys = determine_param_idx(state_dict, layer_num, keys=True)
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
  return state_dict, freeze_idx

def simple_cnn_freeze(source_dataset, target_dataset):
  for train_filepath, val_filepath in getattr(config, f"{target_dataset}_TRAIN_VAL_SET_FILEPATHS"):
    subject = subject_re.findall(val_filepath)[0]
    results = []
    for freeze in range(5):
      name = f"{source_dataset}-{target_dataset}-Simple_CNN-{subject}-FREEZE"
      config_dict = getattr(config, target_dataset).copy()
      config_dict["NAME"] = f"{name}-{freeze}"
      config_dict["MODEL"] = "Simple_CNN"
      config_dict["TRAIN_SET_FILEPATH"] = train_filepath
      config_dict["VAL_SET_FILEPATH"] = val_filepath
      config_dict["LEARNING_RATE"] = config_dict["Simple_CNN_LEARNING_RATE"]
      config_dict["FREEZE"] = freeze
      state_dict, freeze_idx = base_transfer(source_dataset, target_dataset, "Simple_CNN", freeze=freeze)
      print(f"-----{config_dict['NAME']}-----")
      trainer = Trainer(config_dict, state_dict, freeze_idx)
      eval_dict = trainer.train()
      results.append(eval_dict)
      eval_dict_acc, eval_dict_wf1 = eval_dict
      print("ACC: ", eval_dict_acc["best_val"], f"epoch: {eval_dict_acc['best_epoch']}, iteration: {eval_dict_acc['best_iteration']}\n")
      print("WF1: ", eval_dict_wf1["best_val"], f"epoch: {eval_dict_wf1['best_epoch']}, iteration: {eval_dict_wf1['best_iteration']}\n")
    save_best_result(results, name, "FREEZE")

def simple_cnn_layer_num(source_dataset, target_dataset):
  for train_filepath, val_filepath in getattr(config, f"{target_dataset}_TRAIN_VAL_SET_FILEPATHS"):
    subject = subject_re.findall(val_filepath)[0]
    results = []
    for layer_num in range(1,4):
      name = f"{source_dataset}-{target_dataset}-Simple_CNN-{subject}-LAYER_NUM"
      config_dict = getattr(config, target_dataset).copy()
      config_dict["NAME"] = f"{name}-{layer_num}"
      config_dict["MODEL"] = "Simple_CNN"
      config_dict["TRAIN_SET_FILEPATH"] = train_filepath
      config_dict["VAL_SET_FILEPATH"] = val_filepath
      config_dict["LEARNING_RATE"] = config_dict["Simple_CNN_LEARNING_RATE"]
      config_dict["LAYER_NUM"] = layer_num
      state_dict, freeze_idx = base_transfer(source_dataset, target_dataset, "Simple_CNN", layer_num=layer_num)
      print(f"-----{config_dict['NAME']}-----")
      trainer = Trainer(config_dict, state_dict, freeze_idx)
      eval_dict = trainer.train()
      results.append(eval_dict)
      eval_dict_acc, eval_dict_wf1 = eval_dict
      print("ACC: ", eval_dict_acc["best_val"], f"epoch: {eval_dict_acc['best_epoch']}, iteration: {eval_dict_acc['best_iteration']}\n")
      print("WF1: ", eval_dict_wf1["best_val"], f"epoch: {eval_dict_wf1['best_epoch']}, iteration: {eval_dict_wf1['best_iteration']}\n")
    save_best_result(results, name, "LAYER_NUM")

def all_simple_cnn_freeze():
  for source_dataset in ["PAMAP2", "OPPORTUNITY_LOCOMOTION", "OPPORTUNITY_GESTURES"]:
    for target_dataset in ["ORDER_PICKING_A", "ORDER_PICKING_B"]:
      simple_cnn_freeze(source_dataset, target_dataset)

def all_simple_cnn_layer_num():
  for source_dataset in ["PAMAP2", "OPPORTUNITY_LOCOMOTION", "OPPORTUNITY_GESTURES"]:
    for target_dataset in ["ORDER_PICKING_A", "ORDER_PICKING_B"]:
      simple_cnn_layer_num(source_dataset, target_dataset)


if __name__ == "__main__":
  parser = ArgumentParser(description="Start predefined experiments")
  parser.add_argument("experiment", help="the experiment to start")
  parser.add_argument("-s", action="store_true", help="saves all results instead of only the best")
  args = parser.parse_args()
  SAVEALL = args.s
  globals()[args.experiment]()