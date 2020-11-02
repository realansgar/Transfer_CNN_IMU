from argparse import ArgumentParser, FileType
import os
import torch
from torch.utils.data import DataLoader
from config import DEVICE, TEST_BASEPATH
from datasets import HARWindows
import metrics

def log(filepath):
  eval_dict = torch.load(filepath, map_location=DEVICE)
  print({k: v for k, v in eval_dict.items() if k not in ["net", "train", "val"]})

def test(filepath):
  eval_dict = torch.load(filepath, map_location=DEVICE)
  test_set = HARWindows(eval_dict["config"]["TEST_SET_FILEPATH"])
  test_dataloader = DataLoader(test_set, batch_size=len(test_set))
  eval_test = metrics.evaluate_net(eval_dict["net"], torch.nn.CrossEntropyLoss(), next(iter(test_dataloader)), eval_dict["config"]["NUM_CLASSES"])
  print(eval_test)
  torch.save(eval_test, TEST_BASEPATH + os.path.basename(filepath))


if __name__ == "__main__":
  parser = ArgumentParser(description="Display logs and test saved model")
  parser.add_argument("method", choices=["log", "test"], help="the function to call")
  parser.add_argument("files", type=FileType("r"), nargs="*", help="the files to log or test")
  args = parser.parse_args()

  for file in args.files:
    globals()[args.method](file.name)
