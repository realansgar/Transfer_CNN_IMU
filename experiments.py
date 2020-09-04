from train import Trainer
from config import PAMAP2

def pamap2_hyperparameters():
  pamap2_trainer = Trainer(PAMAP2)
  pamap2_trainer.train()