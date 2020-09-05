from argparse import ArgumentParser
import experiments
from config import *

if __name__ == "__main__":
  parser = ArgumentParser(description="Starts experiments and produces plots for those experiments.")
  parser.add_argument("action", choices=["experiment", "plot"], help="what you want to do")
  parser.add_argument("filename")
  args = parser.parse_args()

  if args.action == "experiment":
    experiments.pamap2_hyperparameters()
  elif args.action == "plot":
    experiments.plot(args.filename)