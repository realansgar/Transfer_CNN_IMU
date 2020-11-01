import re
from itertools import chain
from train import Trainer
from config import PAMAP2

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


def pamap2_hyperparameters():
  #pamap2_cnn_imu_2 = PAMAP2.copy()
  #pamap2_trainer = Trainer(pamap2_cnn_imu_2)
  #pamap2_trainer.train()

  pamap2_simple_cnn = PAMAP2.copy()
  pamap2_simple_cnn["NAME"] = "PAMAP2 - SimpleCNN"
  pamap2_simple_cnn["MODEL"] = "SimpleCNN"
  pamap2_trainer = Trainer(pamap2_simple_cnn)
  pamap2_trainer.train()

if __name__ == "__main__":
  # TODO insert argparser for experiments
  pamap2_hyperparameters()