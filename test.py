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

def test(eval_dict, acc=False):
  test_set = HARWindows(eval_dict["config"]["TEST_SET_FILEPATH"])
  test_dataloader = DataLoader(test_set, batch_size=len(test_set))
  eval_test = metrics.evaluate_net(eval_dict["net"], torch.nn.CrossEntropyLoss(), next(iter(test_dataloader)), eval_dict["config"]["NUM_CLASSES"])
  eval_dict["test"] = eval_test
  print(eval_dict["config"]["NAME"], eval_test, "\n")
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
    result_dict_acc = {k: [eval_dict["test"][k].cpu() for eval_dict in results_acc] for k in results_acc[0]["test"]}
    result_dict_acc_mean = {f"{k}_mean": np.mean(v) for k, v in result_dict_acc.items()}
    result_dict_acc_conf = ({f"{k}_conf": np.mean(v) - st.t.interval(0.95, len(v)-1, loc=np.mean(v), scale=st.sem(v))[0] for k, v in result_dict_acc.items()})
    result_dict_acc.update(result_dict_acc_mean)
    result_dict_acc.update(result_dict_acc_conf)
    torch.save(result_dict_acc, f"{config.TEST_BASEPATH}{name}_results_acc_{result_dict_acc['micro_accuracy_mean']}.pt")
    result_dict_wf1 = {k: [eval_dict["test"][k].cpu() for eval_dict in results_wf1] for k in results_wf1[0]["test"]}
    result_dict_wf1_mean = {f"{k}_mean": np.mean(v) for k, v in result_dict_wf1.items()}
    result_dict_wf1_conf = ({f"{k}_conf": np.mean(v) - st.t.interval(0.95, len(v)-1, loc=np.mean(v), scale=st.sem(v))[0] for k, v in result_dict_wf1.items()})
    result_dict_wf1.update(result_dict_wf1_mean)
    result_dict_wf1.update(result_dict_wf1_conf)
    torch.save(result_dict_wf1, f"{config.TEST_BASEPATH}{name}_results_wf1_{result_dict_wf1['weighted_f1_mean']}.pt")

def test_config_order_picking(dataset):
  config.DETERMINISTIC = False
  for train_filepath, val_filepath in getattr(config, f"{dataset}_TRAIN_VAL_SET_FILEPATHS"):
    subject = subject_re.findall(val_filepath)[0]
    for model in ["Simple_CNN", "CNN_IMU"]:
      name = f"{dataset}-{subject}"
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
        eval_dict_wf1 = test(eval_dict_wf1)
        eval_dict_acc = test(eval_dict_acc, acc=True)
        results_acc.append(eval_dict_acc)
        results_wf1.append(eval_dict_wf1)
      result_dict_acc = {k: [eval_dict["test"][k].cpu() for eval_dict in results_acc] for k in results_acc[0]["test"]}
      result_dict_acc_mean = {f"{k}_mean": np.mean(v) for k, v in result_dict_acc.items()}
      result_dict_acc_conf = ({f"{k}_conf": np.mean(v) - st.t.interval(0.95, len(v)-1, loc=np.mean(v), scale=st.sem(v))[0] for k, v in result_dict_acc.items()})
      result_dict_acc.update(result_dict_acc_mean)
      result_dict_acc.update(result_dict_acc_conf)
      torch.save(result_dict_acc, f"{config.TEST_BASEPATH}{name}_results_acc_{result_dict_acc['micro_accuracy_mean']}.pt")
      result_dict_wf1 = {k: [eval_dict["test"][k].cpu() for eval_dict in results_wf1] for k in results_wf1[0]["test"]}
      result_dict_wf1_mean = {f"{k}_mean": np.mean(v) for k, v in result_dict_wf1.items()}
      result_dict_wf1_conf = ({f"{k}_conf": np.mean(v) - st.t.interval(0.95, len(v)-1, loc=np.mean(v), scale=st.sem(v))[0] for k, v in result_dict_wf1.items()})
      result_dict_wf1.update(result_dict_wf1_mean)
      result_dict_wf1.update(result_dict_wf1_conf)
      torch.save(result_dict_wf1, f"{config.TEST_BASEPATH}{name}_results_wf1_{result_dict_wf1['weighted_f1_mean']}.pt")

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
    for file in args.files:
      globals()[args.method](file.name)
