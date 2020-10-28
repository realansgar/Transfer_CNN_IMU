from argparse import ArgumentParser
import numpy as np
import json
from config import *
from sliding_window import sliding_window

def delete_labels(data, label_mask):
  """
  Deletes all rows with labels not in label_mask from data

  :param data: 2darray with shape (rows, columns)
  :param label_mask: iterable with label values that should not be deleted

  :returns 2darray with rows deleted 
  """

  idl = [np.where(data[:,0] == i)[0] for i in label_mask]
  idl = np.concatenate(idl)                                                 # indexes of all rows with a label that is in label_mask
  mask = np.ones(data.shape[0], dtype=bool)
  mask[idl] = False
  return np.delete(data, mask, axis=0)                                      # delete all rows with a label not selected

def remap_labels(data, label_remapping):
  """
  Remaps labels to eliminate gaps between label ids

  :param data: 2darray with shape (rows, columns)
  :param label_remapping: dict with shape {old_label: new_label}

  :returns 2darray with labels remapped
  """

  for old_label, new_label in label_remapping.items():
    idl = np.where(data[:,0] == old_label)[0]
    data[idl,0] = new_label
  return data

def normalize_data(data, min_list, max_list):
  """
  Normalizes sensor channels to a range [0,1]

  :param data: 2darray with shape (rows, columns)
  :param min_list: list of minimum values for every data column
  :param max_list: list of maximum values for every data column
  :returns 2darray with normalized data
  """

  min_list, max_list = np.array(min_list), np.array(max_list)
  data[:,1:] = (data[:,1:] - min_list) / (max_list - min_list)
  return data

def downsample_data(data, current_freq, target_freq):
  """
  Downsample time series data to a target frequency. Currently not an accurate implementation!

  :param data: 2darray with shape (rows, columns)
  :param current_freq: time series frequency of data
  :param target_freq: targeted time series frequency
  :returns 2darray with every (current_freq // target_freq)-th row
  """
  if target_freq > current_freq:
    raise ValueError(f"current_freq: {current_freq} is less than target_freq: {target_freq}")
  downsample_rate = current_freq // target_freq
  return data[::downsample_rate]

def split_data(data):
  """
  Split data into data and label portion

  :param data: 2darray with shape (rows, columns)
  :returns tupel (data_x, data_y)
  """
  return data[:,1:], data[:,0]


def compute_min_max_json(preprocess_func, filepath):
  print("computing min_max.json ...")
  min_list, max_list = np.PINF, np.NINF
  for data in preprocess_func(normalize=False):
    data_x, _ = split_data(data)
    min_list_cand, max_list_cand = np.amin(data_x, axis=0), np.amax(data_x, axis=0)
    min_list, max_list = np.minimum(min_list, min_list_cand), np.maximum(max_list, max_list_cand)
  min_max_dict = {"min_list": list(min_list), "max_list": list(max_list)}
  with open(filepath, "w") as min_max_json:
    json.dump(min_max_dict, min_max_json)
    print(f"saved {filepath}")

def build_set(filepath, config):
  print(f"generating windows from {filepath}")
  data = np.load(filepath)
  data_x, data_y = split_data(data)
  data_y = data_y.astype(int)

  data_x_windows = sliding_window(data_x, (config["WINDOW_SIZE"], data_x.shape[1]), (config["STEP_SIZE"], 1))

  data_y_windows = sliding_window(data_y, config["WINDOW_SIZE"], config["STEP_SIZE"])
  data_y_windows = [np.argmax(np.bincount(window, minlength=config["NUM_CLASSES"])) for window in data_y_windows]
  data_y_windows = np.array(data_y_windows)

  return data_x_windows, data_y_windows

def build_train_val_test_set(config):
  x_train_set, x_val_set, x_test_set = [], [], []
  y_train_set, y_val_set, y_test_set = [], [], []

  print("building train_set")
  for filepath in config["TRAIN_SET"]:
    x_windows, y_windows = build_set(filepath, config)
    x_train_set.append(x_windows)
    y_train_set.append(y_windows)
  x_train_set, y_train_set = np.concatenate(x_train_set, axis=0), np.concatenate(y_train_set, axis=0)
  np.savez(config["TRAIN_SET_FILEPATH"], data_x=x_train_set, data_y=y_train_set)
  print(f"saved train_set to {config['TRAIN_SET_FILEPATH']}")

  print("builing val_set")
  for filepath in config["VAL_SET"]:
    x_windows, y_windows = build_set(filepath, config)
    x_val_set.append(x_windows)
    y_val_set.append(y_windows)
  x_val_set, y_val_set = np.concatenate(x_val_set, axis=0), np.concatenate(y_val_set, axis=0)
  np.savez(config["VAL_SET_FILEPATH"], data_x=x_val_set, data_y=y_val_set)
  print(f"saved val_set to {config['VAL_SET_FILEPATH']}")

  print("building test_set")
  for filepath in config["TEST_SET"]:
    x_windows, y_windows = build_set(filepath, config)
    x_test_set.append(x_windows)
    y_test_set.append(y_windows)
  x_test_set, y_test_set = np.concatenate(x_test_set, axis=0), np.concatenate(y_test_set, axis=0)
  np.savez(config["TEST_SET_FILEPATH"], data_x=x_test_set, data_y=y_test_set)
  print(f"saved test_set to {config['TEST_SET_FILEPATH']}")

def preprocess_OPPORTUNITY_LOCOMOTION(normalize=True, write=False):
  for filepath in OPPORTUNITY_LOCOMOTION_FILEPATHS:
    print(f"processing {filepath} ...")
    data = np.loadtxt(filepath)
    data = data[:,OPPORTUNITY_LOCOMOTION_COLUMN_MASK]
    data = delete_labels(data, OPPORTUNITY_LOCOMOTION_LABEL_MASK)
    data = remap_labels(data, OPPORTUNITY_LOCOMOTION_LABEL_REMAPPING)
    data[np.isnan(data)] = 0
    if normalize:
      with open(OPPORTUNITY_LOCOMOTION_MIN_MAX_FILEPATH) as min_max_json:
        min_max_dict = json.load(min_max_json)
        data = normalize_data(data, **min_max_dict)
    if write:
      out_filepath = filepath + OPPORTUNITY_LOCOMOTION_PREPROCESSED_FILENAME_SUFFIX
      np.save(out_filepath, data)
      print(f"saved {out_filepath}")
    yield data

def preprocess_PAMAP2(normalize=True, write=False):
  for filepath in PAMAP2_FILEPATHS:
    print(f"processing {filepath} ...")
    data = np.loadtxt(filepath)                             
    data = np.delete(data, np.invert(PAMAP2_COLUMN_MASK), axis=1)
    data = delete_labels(data, PAMAP2_LABEL_MASK)
    data = downsample_data(data, PAMAP2_CURRENT_FREQUENCY, PAMAP2_TARGET_FREQUENCY)
    data = remap_labels(data, PAMAP2_LABEL_REMAPPING)
    data[np.isnan(data)] = 0
    if normalize:
      with open(PAMAP2_MIN_MAX_FILEPATH) as min_max_json:
        min_max_dict = json.load(min_max_json)
        data = normalize_data(data, **min_max_dict)
    if write:
      out_filepath = filepath + PAMAP2_PREPROCESSED_FILENAME_SUFFIX
      np.save(out_filepath, data)
      print(f"saved {out_filepath}")
    yield data


if __name__ == "__main__":
  parser = ArgumentParser(description="Preprocesses or generates sliding windows according to config.py and min_max.json and saves it to disk.")
  parser.add_argument("dataset", choices=["PAMAP2", "OPPORTUNITY_LOCOMOTION"], help="the dataset to process")
  parser.add_argument("action", choices=["all", "min_max", "preprocess", "windows"], metavar="action", help="all: do the whole pipeline; min_max: compute the min_max.json; preprocess: preprocess the dataset with min_max.json; windows: generate sliding windows from the preprocess dataset")
  args = parser.parse_args()

  locals_dict = locals()
  preprocess_f = locals_dict[f"preprocess_{args.dataset}"]

  if args.action in ["all", "min_max"]:
    min_max_filepath = locals_dict[f"{args.dataset}_MIN_MAX_FILEPATH"]
    compute_min_max_json(preprocess_f, min_max_filepath)
  
  if args.action in ["all", "preprocess"]:
    for _ in preprocess_f(write=True):
      pass

  if args.action in ["all", "windows"]:
    locals_dict["build_train_val_test_set"](locals_dict[args.dataset])
