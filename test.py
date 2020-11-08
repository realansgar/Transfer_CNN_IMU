from argparse import ArgumentParser, FileType
import os
import re
import numpy as np
from scipy import stats as st
import torch
from torch.utils.data import DataLoader
from train import Trainer
import config
from datasets import HARWindows
import metrics

subject_re = re.compile(r"subject\d\d\d")

def log(filepath):
  eval_dict = torch.load(filepath, map_location=config.DEVICE)
  print({k: v for k, v in eval_dict.items() if k not in ["net", "train", "val"]})

def test_filepath(filepath):
  eval_dict = torch.load(filepath, map_location=config.DEVICE)
  test(eval_dict)

def test(eval_dict):
  test_set = HARWindows(eval_dict["config"]["TEST_SET_FILEPATH"])
  test_dataloader = DataLoader(test_set, batch_size=len(test_set))
  eval_test = metrics.evaluate_net(eval_dict["net"], torch.nn.CrossEntropyLoss(), next(iter(test_dataloader)), eval_dict["config"]["NUM_CLASSES"])
  eval_dict["test"] = eval_test
  print(eval_dict["config"]["NAME"], eval_test)
  os.makedirs(config.TEST_BASEPATH, exist_ok=True)
  torch.save(eval_dict, f"{config.TEST_BASEPATH}{eval_dict['config']['NAME']}_wf1_{eval_test['weighted_f1']:.4f}.pt")
  return eval_dict

def test_config(dataset):
  for model in ["Simple_CNN", "CNN_IMU"]:
    results = []
    for i in range(config.TEST_REPETITIONS):
      config_dict = getattr(config, dataset).copy()
      name = config_dict["NAME"]
      config_dict["NAME"] = f"{name}-{i}"
      config_dict["MODEL"] = model
      print(f"-----{config_dict['NAME']}-----")
      trainer = Trainer(config_dict)
      eval_dict = trainer.train()
      _, eval_dict_wf1 = eval_dict
      eval_dict = test(eval_dict_wf1)
      results.append(eval_dict)
      print(eval_dict["test"], "\n")
    result_dict = {k: [eval_dict["test"][k].cpu() for eval_dict in results] for k in results[0]}
    result_dict_mean = {f"{k}_mean": np.mean(v) for k, v in result_dict.items()}
    result_dict_conf = ({f"{k}_conf": np.mean(v) - st.t.interval(0.95, len(v)-1, loc=np.mean(v), scale=st.sem(v))[0] for k, v in result_dict.items()})
    result_dict.update(result_dict_mean)
    result_dict.update(result_dict_conf)
    torch.save(result_dict, f"{config.TEST_BASEPATH}{name}_results.pt")

def test_config_order_picking(dataset):
  for train_filepath, val_filepath in getattr(config, f"{dataset}_TRAIN_VAL_SET_FILEPATHS"):
    subject = subject_re.findall(val_filepath)[0]
    for model in ["Simple_CNN", "CNN_IMU"]:
      name = f"{dataset}-{subject}"
      results = []
      for i in range(config.TEST_REPETITIONS):
        config_dict = getattr(config, dataset).copy()
        config_dict["NAME"] = f"{name}-{i}"
        config_dict["MODEL"] = model
        config_dict["LEARNING_RATE"] = config_dict[f"{model}_LEARNING_RATE"]
        config_dict["TRAIN_SET_FILEPATH"] = train_filepath
        config_dict["VAL_SET_FILEPATH"] = val_filepath
        print(f"-----{config_dict['NAME']}-----")
        trainer = Trainer(config_dict)
        eval_dict = trainer.train()
        _, eval_dict_wf1 = eval_dict
        eval_dict = test(eval_dict_wf1)
        results.append(eval_dict)
        print(eval_dict["test"], "\n")
      result_dict = {k: [eval_dict["test"][k] for eval_dict in results] for k in results[0]}
      result_dict_mean = {f"{k}_mean": np.mean(v) for k, v in result_dict.items()}
      result_dict_conf = ({f"{k}_conf": np.mean(v) - st.t.interval(0.95, len(v)-1, loc=np.mean(v), scale=st.sem(v))[0] for k, v in result_dict.items()})
      result_dict.update(result_dict_mean)
      result_dict.update(result_dict_conf)
      torch.save(result_dict, f"{config.TEST_BASEPATH}{name}_results.pt")

def test_all_learning_rate():
  for dataset in ["PAMAP2", "OPPORTUNITY_LOCOMOTION", "OPPORTUNITY_GESTURES"]:
    test_config(dataset)
  for dataset in ["ORDER_PICKING_A", "ORDER_PICKING_B"]:
    test_config_order_picking(dataset)


if __name__ == "__main__":
  parser = ArgumentParser(description="Display logs and test saved model")
  parser.add_argument("method", help="the function to call")
  parser.add_argument("files", type=FileType("r"), nargs="*", help="the files to log or test")
  args = parser.parse_args()

  for file in args.files:
    globals()[args.method](file.name)
