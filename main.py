from argparse import ArgumentParser
import experiments
from config import *

if __name__ == "__main__":
  parser = ArgumentParser(description="Starts experiments and produces plots for those experiments.")
  
  args = parser.parse_args()

  experiments.pamap2_hyperparameters()