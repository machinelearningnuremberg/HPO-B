from genericpath import exists
import torch
from fsbo_modules import FSBO
import numpy as np
import os
import json
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def main(args):
    
    rootdir     = os.path.dirname(os.path.realpath(__file__))
    np.random.seed(123)
    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 


    path = os.path.join(rootdir,"../..", "data","hpob-data")

    with open(path+"/meta-validation-dataset.json", "r") as f:
        valid_data = json.load(f) 
        valid_data = valid_data[args.space]    

    with open(path+"/meta-train-dataset.json", "r") as f:
        train_data = json.load(f) 
        train_data = train_data[args.space]

    os.makedirs(os.path.join(rootdir,"checkpoints"), exist_ok=True)
    checkpoint_path = os.path.join(rootdir,"checkpoints","FSBO2", f"{args.space}")
    fsbo_model = FSBO(train_data = train_data, valid_data = valid_data, checkpoint_path = checkpoint_path)
    fsbo_model.meta_train(epochs=args.epochs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--space', help='Search Space Id', type=str, default="4796")
    parser.add_argument('--epochs', help='Meta-Train epochs', type=str, default=100000)

    args = parser.parse_args()

    main(args)