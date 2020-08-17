import numpy as np
import pickle
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

def preprocess_PAMAP2_data():
  for pamap2_filepath in PAMAP2_FILEPATHS:
    filepath = DATASETS_BASEPATH + PAMAP2_BASEPATH + pamap2_filepath
    data = np.loadtxt(filepath)                             
    data = np.delete(data, np.invert(PAMAP2_COLUMN_MASK), axis=1)                
    data = delete_labels(data, PAMAP2_LABEL_MASK, PAMAP2_LABEL_COLUMN_INDEX)     
    data = remap_labels(data, PAMAP2_LABEL_REMAPPING, PAMAP2_LABEL_COLUMN_INDEX) 


def compute_max_min(dataset_filepaths, out_filepath):

  pass

filepath = "PAMAP2_Dataset/Protocol/subject101.dat"



print(data.shape)