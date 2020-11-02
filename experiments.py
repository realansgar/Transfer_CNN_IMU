import re
from itertools import chain
import os
from argparse import ArgumentParser
import torch
from train import Trainer
from config import *

subject_re = re.compile(r"subject\d\d\d")

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
  simple_cnn_results = []
  for epochs in [20,12,8]:
    config = PAMAP2.copy()
    config["NAME"] = f"PAMAP2-Simple_CNN-{epochs}ep"
    config["MODEL"] = "Simple_CNN"
    config["EPOCHS"] = epochs
    trainer = Trainer(config)
    eval_dict = trainer.train()
    simple_cnn_results.append(eval_dict)
    print(eval_dict["best_val"])
  save_best_result(simple_cnn_results, "PAMAP2-Simple_CNN-ep")

  cnn_imu_results = []
  for epochs in [20,12,8]:
    config = PAMAP2.copy()
    config["NAME"] = f"PAMAP2-CNN_IMU-{epochs}ep"
    config["MODEL"] = "CNN_IMU"
    config["EPOCHS"] = epochs
    trainer = Trainer(config)
    eval_dict = trainer.train()
    cnn_imu_results.append(eval_dict)
    print(eval_dict["best_val"])
  save_best_result(cnn_imu_results, "PAMAP2-CNN_IMU-ep")

def pamap2_learning_rate():
  simple_cnn_results = []
  for lr in [10**-3, 10**-4, 10**-5]:
    config = PAMAP2.copy()
    config["NAME"] = f"PAMAP2-Simple_CNN-{lr}lr"
    config["MODEL"] = "Simple_CNN"
    config["LEARNING_RATE"] = lr
    trainer = Trainer(config)
    eval_dict = trainer.train()
    simple_cnn_results.append(eval_dict)
    print(eval_dict["best_val"])
  save_best_result(simple_cnn_results, "PAMAP2-Simple_CNN-lr")

  cnn_imu_results = []
  for lr in [10**-3, 10**-4, 10**-5]:
    config = PAMAP2.copy()
    config["NAME"] = f"PAMAP2-CNN_IMU-{lr}lr"
    config["MODEL"] = "CNN_IMU"
    config["LEARNING_RATE"] = lr
    trainer = Trainer(config)
    eval_dict = trainer.train()
    cnn_imu_results.append(eval_dict)
    print(eval_dict["best_val"])
  save_best_result(cnn_imu_results, "PAMAP2-CNN_IMU-lr")

def opportunity_locomotion_epochs():
  simple_cnn_results = []
  for epochs in [20,12,8]:
    config = OPPORTUNITY_LOCOMOTION.copy()
    config["NAME"] = f"OPPORTUNITY_LOCOMOTION-Simple_CNN-{epochs}ep"
    config["MODEL"] = "Simple_CNN"
    config["EPOCHS"] = epochs
    trainer = Trainer(config)
    eval_dict = trainer.train()
    simple_cnn_results.append(eval_dict)
    print(eval_dict["best_val"])
  save_best_result(simple_cnn_results, "OPPORTUNITY_LOCOMOTION-Simple_CNN-ep")

  cnn_imu_results = []
  for epochs in [20,12,8]:
    config = OPPORTUNITY_LOCOMOTION.copy()
    config["NAME"] = f"OPPORTUNITY_LOCOMOTION-CNN_IMU-{epochs}ep"
    config["MODEL"] = "CNN_IMU"
    config["EPOCHS"] = epochs
    trainer = Trainer(config)
    eval_dict = trainer.train()
    cnn_imu_results.append(eval_dict)
    print(eval_dict["best_val"])
  save_best_result(cnn_imu_results, "OPPORTUNITY_LOCOMOTION-CNN_IMU-ep")

def opportunity_locomotion_learning_rate():
  simple_cnn_results = []
  for lr in [10**-3, 10**-4, 10**-5]:
    config = OPPORTUNITY_LOCOMOTION.copy()
    config["NAME"] = f"OPPORTUNITY_LOCOMOTION-Simple_CNN-{lr}lr"
    config["MODEL"] = "Simple_CNN"
    config["LEARNING_RATE"] = lr
    trainer = Trainer(config)
    eval_dict = trainer.train()
    simple_cnn_results.append(eval_dict)
    print(eval_dict["best_val"])
  save_best_result(simple_cnn_results, "OPPORTUNITY_LOCOMOTION-Simple_CNN-lr")

  cnn_imu_results = []
  for lr in [10**-3, 10**-4, 10**-5]:
    config = OPPORTUNITY_LOCOMOTION.copy()
    config["NAME"] = f"OPPORTUNITY_LOCOMOTION-CNN_IMU-{lr}lr"
    config["MODEL"] = "CNN_IMU"
    config["LEARNING_RATE"] = lr
    trainer = Trainer(config)
    eval_dict = trainer.train()
    cnn_imu_results.append(eval_dict)
    print(eval_dict["best_val"])
  save_best_result(simple_cnn_results, "OPPORTUNITY_LOCOMOTION-CNN_IMU-lr")

def opportunity_gestures_epochs():
  simple_cnn_results = []
  for epochs in [20,12,8]:
    config = OPPORTUNITY_GESTURES.copy()
    config["NAME"] = f"OPPORTUNITY_GESTURES-Simple_CNN-{epochs}ep"
    config["MODEL"] = "Simple_CNN"
    config["EPOCHS"] = epochs
    trainer = Trainer(config)
    eval_dict = trainer.train()
    simple_cnn_results.append(eval_dict)
    print(eval_dict["best_val"])
  save_best_result(simple_cnn_results, "OPPORTUNITY_GESTURES-Simple_CNN-ep")

  cnn_imu_results = []
  for epochs in [20,12,8]:
    config = OPPORTUNITY_GESTURES.copy()
    config["NAME"] = f"OPPORTUNITY_GESTURES-CNN_IMU-{epochs}ep"
    config["MODEL"] = "CNN_IMU"
    config["EPOCHS"] = epochs
    trainer = Trainer(config)
    eval_dict = trainer.train()
    cnn_imu_results.append(eval_dict)
    print(eval_dict["best_val"])
  save_best_result(cnn_imu_results, "OPPORTUNITY_GESTURES-CNN_IMU-ep")

def opportunity_gestures_learning_rate():
  simple_cnn_results = []
  for lr in [10**-3, 10**-4, 10**-5]:
    config = OPPORTUNITY_GESTURES.copy()
    config["NAME"] = f"OPPORTUNITY_GESTURES-Simple_CNN-{lr}lr"
    config["MODEL"] = "Simple_CNN"
    config["LEARNING_RATE"] = lr
    trainer = Trainer(config)
    eval_dict = trainer.train()
    simple_cnn_results.append(eval_dict)
    print(eval_dict["best_val"])
  save_best_result(simple_cnn_results, "OPPORTUNITY_GESTURES-Simple_CNN-lr")

  cnn_imu_results = []
  for lr in [10**-3, 10**-4, 10**-5]:
    config = OPPORTUNITY_GESTURES.copy()
    config["NAME"] = f"OPPORTUNITY_GESTURES-CNN_IMU-{lr}lr"
    config["MODEL"] = "CNN_IMU"
    config["LEARNING_RATE"] = lr
    trainer = Trainer(config)
    eval_dict = trainer.train()
    cnn_imu_results.append(eval_dict)
    print(eval_dict["best_val"])
  save_best_result(simple_cnn_results, "OPPORTUNITY_GESTURES-CNN_IMU-lr")

def order_picking_a_epochs():
  for train_filepath, val_filepath in ORDER_PICKING_A_TRAIN_VAL_SET_FILEPATHS:
    subject = subject_re.findall(val_filepath)[0]
    simple_cnn_results = []
    for epochs in [25,20,15]:
      config = ORDER_PICKING_A.copy()
      config["TRAIN_SET_FILEPATH"] = train_filepath
      config["VAL_SET_FILEPATH"] = val_filepath
      config["NAME"] = f"ORDER_PICKING_A-Simple_CNN-{subject}-{epochs}ep"
      config["MODEL"] = "Simple_CNN"
      config["EPOCHS"] = epochs
      trainer = Trainer(config)
      eval_dict = trainer.train()
      simple_cnn_results.append(eval_dict)
      print(eval_dict["best_val"])
    save_best_result(simple_cnn_results, f"ORDER_PICKING_A-Simple_CNN-{subject}-ep")

  for train_filepath, val_filepath in ORDER_PICKING_A_TRAIN_VAL_SET_FILEPATHS:
    subject = subject_re.findall(val_filepath)[0]
    cnn_imu_results = []
    for epochs in [25,20,15]:
      config = ORDER_PICKING_A.copy()
      config["TRAIN_SET_FILEPATH"] = train_filepath
      config["VAL_SET_FILEPATH"] = val_filepath
      config["NAME"] = f"ORDER_PICKING_A-CNN_IMU-{subject}-{epochs}ep"
      config["MODEL"] = "CNN_IMU"
      config["EPOCHS"] = epochs
      trainer = Trainer(config)
      eval_dict = trainer.train()
      cnn_imu_results.append(eval_dict)
      print(eval_dict["best_val"])
    save_best_result(cnn_imu_results, f"ORDER_PICKING_A-CNN_IMU-{subject}-ep")

def order_picking_a_learning_rate():
  for train_filepath, val_filepath in ORDER_PICKING_A_TRAIN_VAL_SET_FILEPATHS:
    subject = subject_re.findall(val_filepath)[0]
    simple_cnn_results = []
    for lr in [10**-4, 10**-5, 10**-6, 10**-7]:
      config = ORDER_PICKING_A.copy()
      config["TRAIN_SET_FILEPATH"] = train_filepath
      config["VAL_SET_FILEPATH"] = val_filepath
      config["NAME"] = f"ORDER_PICKING_A-Simple_CNN-{subject}-{lr}lr"
      config["MODEL"] = "Simple_CNN"
      config["LEARNING_RATE"] = lr
      trainer = Trainer(config)
      eval_dict = trainer.train()
      simple_cnn_results.append(eval_dict)
      print(eval_dict["best_val"])
    save_best_result(simple_cnn_results, f"ORDER_PICKING_A-Simple_CNN-{subject}-lr")

  for train_filepath, val_filepath in ORDER_PICKING_A_TRAIN_VAL_SET_FILEPATHS:
    subject = subject_re.findall(val_filepath)[0]
    cnn_imu_results = []
    for lr in [10**-4, 10**-5, 10**-6, 10**-7]:
      config = ORDER_PICKING_A.copy()
      config["TRAIN_SET_FILEPATH"] = train_filepath
      config["VAL_SET_FILEPATH"] = val_filepath      
      config["NAME"] = f"ORDER_PICKING_A-CNN_IMU-{subject}-{lr}lr"
      config["MODEL"] = "CNN_IMU"
      config["LEARNING_RATE"] = lr
      trainer = Trainer(config)
      eval_dict = trainer.train()
      cnn_imu_results.append(eval_dict)
      print(eval_dict["best_val"])
    save_best_result(simple_cnn_results, f"ORDER_PICKING_A-CNN_IMU-{subject}-lr")

def order_picking_b_epochs():
  for train_filepath, val_filepath in ORDER_PICKING_B_TRAIN_VAL_SET_FILEPATHS:
    subject = subject_re.findall(val_filepath)[0]
    simple_cnn_results = []
    for epochs in [25,20,15]:
      config = ORDER_PICKING_B.copy()
      config["TRAIN_SET_FILEPATH"] = train_filepath
      config["VAL_SET_FILEPATH"] = val_filepath
      config["NAME"] = f"ORDER_PICKING_B-Simple_CNN-{subject}-{epochs}ep"
      config["MODEL"] = "Simple_CNN"
      config["EPOCHS"] = epochs
      trainer = Trainer(config)
      eval_dict = trainer.train()
      simple_cnn_results.append(eval_dict)
      print(eval_dict["best_val"])
    save_best_result(simple_cnn_results, f"ORDER_PICKING_B-Simple_CNN-{subject}-ep")

  for train_filepath, val_filepath in ORDER_PICKING_B_TRAIN_VAL_SET_FILEPATHS:
    subject = subject_re.findall(val_filepath)[0]
    cnn_imu_results = []
    for epochs in [25,20,15]:
      config = ORDER_PICKING_B.copy()
      config["TRAIN_SET_FILEPATH"] = train_filepath
      config["VAL_SET_FILEPATH"] = val_filepath
      config["NAME"] = f"ORDER_PICKING_B-CNN_IMU-{subject}-{epochs}ep"
      config["MODEL"] = "CNN_IMU"
      config["EPOCHS"] = epochs
      trainer = Trainer(config)
      eval_dict = trainer.train()
      cnn_imu_results.append(eval_dict)
      print(eval_dict["best_val"])
    save_best_result(cnn_imu_results, f"ORDER_PICKING_B-CNN_IMU-{subject}-ep")

def order_picking_b_learning_rate():
  for train_filepath, val_filepath in ORDER_PICKING_B_TRAIN_VAL_SET_FILEPATHS:
    subject = subject_re.findall(val_filepath)[0]
    simple_cnn_results = []
    for lr in [10**-4, 10**-5, 10**-6, 10**-7]:
      config = ORDER_PICKING_B.copy()
      config["TRAIN_SET_FILEPATH"] = train_filepath
      config["VAL_SET_FILEPATH"] = val_filepath
      config["NAME"] = f"ORDER_PICKING_B-Simple_CNN-{subject}-{lr}lr"
      config["MODEL"] = "Simple_CNN"
      config["LEARNING_RATE"] = lr
      trainer = Trainer(config)
      eval_dict = trainer.train()
      simple_cnn_results.append(eval_dict)
      print(eval_dict["best_val"])
    save_best_result(simple_cnn_results, f"ORDER_PICKING_B-Simple_CNN-{subject}-lr")

  for train_filepath, val_filepath in ORDER_PICKING_B_TRAIN_VAL_SET_FILEPATHS:
    subject = subject_re.findall(val_filepath)[0]
    cnn_imu_results = []
    for lr in [10**-4, 10**-5, 10**-6, 10**-7]:
      config = ORDER_PICKING_B.copy()
      config["TRAIN_SET_FILEPATH"] = train_filepath
      config["VAL_SET_FILEPATH"] = val_filepath      
      config["NAME"] = f"ORDER_PICKING_B-CNN_IMU-{subject}-{lr}lr"
      config["MODEL"] = "CNN_IMU"
      config["LEARNING_RATE"] = lr
      trainer = Trainer(config)
      eval_dict = trainer.train()
      cnn_imu_results.append(eval_dict)
      print(eval_dict["best_val"])
    save_best_result(simple_cnn_results, f"ORDER_PICKING_B-CNN_IMU-{subject}-lr")


if __name__ == "__main__":
  parser = ArgumentParser(description="Start predefined experiments")
  parser.add_argument("experiment", help="the experiment to start")
  args = parser.parse_args()
  globals()[args.experiment]()