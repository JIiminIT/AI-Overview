import argparse
import os
import random
import numpy as np
import torch
from train import run_training
from inference import run_inference
import wandb

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str, default='UCI_HAR_Project', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='resnet_transformer_run1', help='wandb run name')
    parser.add_argument('--data_dir', type=str, default='./dataset/UCI HAR Dataset')
    parser.add_argument('--save_dir', type=str, default='./model_save/UCI_model', help='Path for saving model')
    parser.add_argument('--model_path', type=str, default='best_model.pth')
    parser.add_argument('--best_acc', type=int, default=0.8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--inference', action='store_true',default=False)
    parser.add_argument('--class_labels', nargs='+', default=[
        'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS',
        'SITTING', 'STANDING', 'LAYING'
    ])
    args = parser.parse_args()

    set_seed(42)

    print(f"Using device: {args.device}")
    if args.inference:
        print("Inference mode ON")
        run_inference(args)
    else:
        print("Training mode ON")
        run_training(args)
        run_inference(args)


if __name__ == '__main__':
    main()
