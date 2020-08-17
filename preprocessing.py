from argparse import ArgumentParser
import numpy as np
import json
from config import *

def delete_labels(data, label_mask, label_column_index):
  """
  Deletes all rows with labels not in label_mask from data

  :param data: 2darray with shape (rows, columns)
  :param label_mask: iterable with label values that should not be deleted
  :param label_column_index: index of the label column in data 

  :returns 2darray with rows deleted 
  """

  idl = [np.where(data[:,label_column_index] == i)[0] for i in label_mask]
  idl = np.concatenate(idl)                                                 # indexes of all rows with a label that is in label_mask
  mask = np.ones(data.shape[0], dtype=bool)
  mask[idl] = False
  return np.delete(data, mask, axis=0)                                      # delete all rows with a label not selected

def remap_labels(data, label_remapping, label_column_index):
  """
  Remaps labels to eliminate gaps between label ids

  :param data: 2darray with shape (rows, columns)
  :param label_remapping: dict with shape {old_label: new_label}
  :param label_column_index: index of label column in data

  :returns 2darray with labels remapped
  """

  for old_label, new_label in label_remapping.items():
    idl = np.where(data[:,label_column_index] == old_label)[0]
    data[idl,label_column_index] = new_label
  return data

def normalize_data(data, data_column_index, min_list, max_list):
  """
  Normalizes sensor channels to a range [0,1]

  :param data: 2darray with shape (rows, columns)
  :param data_column_index: columns is of shape [..., not data, data, ...] data starts at data_column_index
  :param min_list: list of minimum values for every data column
  :param max_list: list of maximum values for every data column
  :returns 2darray with normalized data
  """

  min_list, max_list = np.array(min_list), np.array(max_list)
  data[:,data_column_index:] = (data[:,data_column_index:] - min_list) / (max_list - min_list)
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


def compute_min_max_json(data_column_index, filepath):
  print("computing min_max.json ...")
  min_list, max_list = np.PINF, np.NINF
  for data in preprocess_PAMAP2(normalize=False):
    min_list_cand, max_list_cand = np.amin(data[:,data_column_index:], axis=0), np.amax(data[:,data_column_index:], axis=0)
    min_list, max_list = np.minimum(min_list, min_list_cand), np.maximum(max_list, max_list_cand)
  min_max_dict = {"min_list": list(min_list), "max_list": list(max_list)}
  with open(filepath, "w") as min_max_json:
    json.dump(min_max_dict, min_max_json)
    print(f"saved {filepath}")


def preprocess_PAMAP2(normalize=True, write=False):
  for filepath in PAMAP2_FILEPATHS:
    print(f"processing {filepath} ...")
    data = np.loadtxt(filepath)                             
    data = np.delete(data, np.invert(PAMAP2_COLUMN_MASK), axis=1)
    data = delete_labels(data, PAMAP2_LABEL_MASK, PAMAP2_LABEL_COLUMN_INDEX)
    data = downsample_data(data, PAMAP2_CURRENT_FREQUENCY, PAMAP2_TARGET_FREQUENCY)
    data = remap_labels(data, PAMAP2_LABEL_REMAPPING, PAMAP2_LABEL_COLUMN_INDEX)
    data[np.isnan(data)] = 0
    if normalize:
      with open(PAMAP2_MIN_MAX_FILEPATH) as min_max_json:
        min_max_dict = json.load(min_max_json)
        data = normalize_data(data, PAMAP2_DATA_COLUMN_INDEX, **min_max_dict)
    if write:
      out_filepath = filepath + PAMAP2_PREPROCESSED_FILENAME_SUFFIX
      np.save(out_filepath, data)
      print(f"saved {filepath}")
    yield data


if __name__ == "__main__":
  parser = ArgumentParser(description="Preprocess a dataset according to config.py and min_max.json and saves it to disk.")
  parser.add_argument("dataset", choices=["PAMAP2"])
  parser.add_argument("--min_max", help="only computes the min_max.json file", action="store_true")
  parser.add_argument("--all", help="computes min_max.json and then preprocesses", action="store_true")
  args = parser.parse_args()

  locals_dict = locals()

  if args.all or args.min_max:
    data_column_index = locals_dict[f"{args.dataset}_DATA_COLUMN_INDEX"]
    min_max_filepath = locals_dict[f"{args.dataset}_MIN_MAX_FILEPATH"]
    compute_min_max_json(data_column_index, min_max_filepath)
  
  if args.min_max:
    quit()
  
  preprocess_func = locals_dict[f"preprocess_{args.dataset}"]
  for data in preprocess_func(write=True):
    pass
