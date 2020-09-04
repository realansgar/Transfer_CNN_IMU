from argparse import ArgumentParser
from train import Trainer
from config import *

if __name__ == "__main__":
  parser = ArgumentParser(description="Starts experiments and produces plots for those experiments.")
  
  args = parser.parse_args()