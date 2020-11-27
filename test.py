from argparse import ArgumentParser, FileType
import os
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from train import Trainer
import config
from datasets import HARWindows
import metrics

subject_re = re.compile(r"subject\d\d\d")

def log(filepaths):
  for filepath in filepaths:
    eval_dict = torch.load(filepath, map_location=config.DEVICE)
    val_set = HARWindows(eval_dict["config"]["VAL_SET_FILEPATH"])
    val_dataloader = DataLoader(val_set, batch_size=len(val_set))
    eval_val = metrics.evaluate_net(eval_dict["net"], torch.nn.CrossEntropyLoss(), next(iter(val_dataloader)), eval_dict["config"]["NUM_CLASSES"])
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    print(eval_dict["config"]["NAME"])
    print(f"wF1: {eval_dict['best_val']['weighted_f1']}")
    print(eval_val["confusion"])

def test_filepath(filepaths):
  for filepath in filepaths:
    eval_dict = torch.load(filepath, map_location=config.DEVICE)
    test(eval_dict)

def test(eval_dict, acc=False):
  test_set = HARWindows(eval_dict["config"]["TEST_SET_FILEPATH"])
  test_dataloader = DataLoader(test_set, batch_size=len(test_set))
  eval_test = metrics.evaluate_net(eval_dict["net"], torch.nn.CrossEntropyLoss(), next(iter(test_dataloader)), eval_dict["config"]["NUM_CLASSES"])
  eval_dict["test"] = eval_test
  print(eval_dict["config"]["NAME"], "ACC" if acc else "WF1", eval_test, "\n")
  os.makedirs(config.TEST_BASEPATH, exist_ok=True)
  if acc:
    torch.save(eval_dict, f"{config.TEST_BASEPATH}{eval_dict['config']['NAME']}_acc_{eval_test['micro_accuracy']:.4f}.pt")
  else:
    torch.save(eval_dict, f"{config.TEST_BASEPATH}{eval_dict['config']['NAME']}_wf1_{eval_test['weighted_f1']:.4f}.pt")
  return eval_dict

def test_config(dataset):
  config.DETERMINISTIC = False
  for model in ["Simple_CNN", "CNN_IMU"]:
    results_acc, results_wf1 = [], []
    for i in range(config.TEST_REPETITIONS):
      config_dict = getattr(config, dataset).copy()
      name = f"{dataset}-{model}"
      config_dict["NAME"] = f"{name}-{i}"
      config_dict["MODEL"] = model
      print(f"-----{config_dict['NAME']}-----")
      trainer = Trainer(config_dict)
      eval_dict = trainer.train()
      eval_dict_acc, eval_dict_wf1 = eval_dict
      eval_dict_wf1 = test(eval_dict_wf1)
      eval_dict_acc = test(eval_dict_acc, acc=True)
      results_acc.append(eval_dict_acc)
      results_wf1.append(eval_dict_wf1)
    result_dict_acc, _, _ = get_aggr_mean_conf(results_acc)
    torch.save(result_dict_acc, f"{config.TEST_BASEPATH}{name}_results_acc_{result_dict_acc['micro_accuracy_mean']}.pt")
    result_dict_wf1, _, _ = get_aggr_mean_conf(results_wf1)
    torch.save(result_dict_wf1, f"{config.TEST_BASEPATH}{name}_results_wf1_{result_dict_wf1['weighted_f1_mean']}.pt")

def test_config_order_picking(dataset):
  config.DETERMINISTIC = False
  subject_results_acc = {"Simple_CNN": [], "CNN_IMU": []}
  subject_results_wf1 = {"Simple_CNN": [], "CNN_IMU": []}
  for train_filepath, val_filepath in getattr(config, f"{dataset}_TRAIN_VAL_SET_FILEPATHS"):
    subject = subject_re.findall(val_filepath)[0]
    for model in ["Simple_CNN", "CNN_IMU"]:
      name = f"{dataset}-{model}-{subject}"
      results_acc, results_wf1 = [], []
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
        eval_dict_acc, eval_dict_wf1 = eval_dict
        print(eval_dict_acc["config"]["NAME"], "ACC", eval_dict_acc["best_val"], "\n")
        print(eval_dict_wf1["config"]["NAME"], "WF1", eval_dict_wf1["best_val"], "\n")
        results_acc.append(eval_dict_acc)
        subject_results_acc[model].append(eval_dict_acc)
        results_wf1.append(eval_dict_wf1)
        subject_results_wf1[model].append(eval_dict_wf1)
      result_dict_acc, _, _ = get_aggr_mean_conf(results_acc, key="best_val")
      torch.save(result_dict_acc, f"{config.TEST_BASEPATH}{name}_results_acc_{result_dict_acc['micro_accuracy_mean']}.pt")
      result_dict_wf1, _, _ = get_aggr_mean_conf(results_wf1, key="best_val")
      torch.save(result_dict_wf1, f"{config.TEST_BASEPATH}{name}_results_wf1_{result_dict_wf1['weighted_f1_mean']}.pt")
  for k, v in subject_results_acc.items():
    _, mean, conf = get_aggr_mean_conf(v, key="best_val")
    mean.update(conf)
    torch.save(mean, f"{config.TEST_BASEPATH}{dataset}-{k}_mean_results_acc_{mean['micro_accuracy_mean']}.pt")
  for k, v in subject_results_wf1.items():
    _, mean, conf = get_aggr_mean_conf(v, key="best_val")
    mean.update(conf)
    torch.save(mean, f"{config.TEST_BASEPATH}{dataset}-{k}_mean_results_wf1_{mean['weighted_f1_mean']}.pt")
  
      

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

  if len(args.files) == 0:
    globals()[args.method]()
  else:
    f_paths = [file.name for file in args.files]
    globals()[args.method](f_paths)
