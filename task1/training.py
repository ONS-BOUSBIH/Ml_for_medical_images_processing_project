import torch
import os
import argparse
from torch.utils.data import DataLoader
from utils_data_loading import *
from utils_training import *
from monai.networks.nets import UNETR
from monai.losses import DiceCELoss
from datetime import datetime
import time

# =====================================
#  File for running the training
# =====================================

def main():
    parser = argparse.ArgumentParser(description="Train a UNETR model")
    parser.add_argument("--model_dir", type=str, default="./unetr_test", help="Directory to save model checkpoints")
    parser.add_argument("--ckpt_file", type=str, default="", help="Checkpoint filename")
    parser.add_argument("--train_split_rate", type=float, default=0.7, help="Training data split ratio")
    parser.add_argument("--val_split_rate", type=float, default=0.2, help="Validation data split ratio")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--data_dir_path", type=str, default="/data/training_data/", help="Path to training data")
    
    args = parser.parse_args()
    
    # Parameters
    model_dir = args.model_dir
    ckpt_file = args.ckpt_file
    train_split_rate = args.train_split_rate
    val_split_rate = args.val_split_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    weight_decay = args.weight_decay
    data_dir_path = args.data_dir_path
    
    # Create a model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Logging setup
    log_file = os.path.join(model_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    log_to_file("Starting Training...\n", log_file)
    log_to_file(f"Learning Rate: {lr}, Batch Size: {batch_size}, Epochs: {num_epochs}, Weight Decay: {weight_decay}\n", log_file)

    # Define Model Parameters
    model_params = {
        "in_channels": 1,
        "out_channels": 2,
        "img_size": (96, 96, 96),
        "feature_size": 16,
        "hidden_size": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "proj_type": "perceptron",
        "norm_name": "instance",
        "res_block": True,
        "dropout_rate": 0.0,
    }
    
    # Log Model Parameters
    log_to_file("\nModel Configuration:", log_file)
    for key, value in model_params.items():
        log_to_file(f"{key}: {value}", log_file)

    # Instantiate Model
    model = UNETR(**model_params)

    # Define Loss and Optimizer
    loss_function = DiceCELoss(include_background=False,to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
      
    # Data Loading
    image_paths, mask_paths = [], []
    for f in sorted(os.listdir(data_dir_path)):
        if 'img' in f:
            idx = f.split('.')[0]
            mask_path = os.path.join(data_dir_path, idx + ".label.nii.gz")
            if os.path.isfile(mask_path):
                image_paths.append(os.path.join(data_dir_path, f))
                mask_paths.append(mask_path)

    # Split data
    train_split_idx = int(train_split_rate * len(image_paths))
    val_split_idx = int(val_split_rate * len(image_paths)) + train_split_idx
    
    training_paths = [image_paths[:train_split_idx], mask_paths[:train_split_idx]]
    validation_paths = [image_paths[train_split_idx:val_split_idx], mask_paths[train_split_idx:val_split_idx]]
    
    # Create datasets
    training_dataset = NiftiDataset(training_paths[0], training_paths[1], training_transforms())
    validation_dataset = NiftiDataset(validation_paths[0], validation_paths[1], validation_transforms())

    # Log Preprocessing Steps
    log_transforms(training_transforms(), log_file, "Training")
    log_transforms(validation_transforms(), log_file, "Validation")
    
    # Create dataloaders
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_multiple_crops, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False,
                                   collate_fn=collate_multiple_crops, num_workers=4, pin_memory=True) #batch size changeng from 4

    # Define Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log_to_file(f"\nUsing Device: {device}\n", log_file)
    torch.cuda.synchronize()  # Ensure all GPU operations are complete
    gpu_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB
    log_to_file(f"Model GPU Memory Usage: {gpu_memory:.2f} MB\n", log_file)

    # Initilize a best dice score variable
    best_dice_score = 0
    total_start = time.time()

    for e in range(1, num_epochs + 1):
        log_to_file(f"\nEpoch {e}/{num_epochs}", log_file)
        train_loss, train_time = train(training_loader, model, optimizer, loss_function, device, log_file)
        val_loss, val_dice_score, val_time = validate(validation_loader, model, loss_function, device, log_file)

        # Save if the new validation dice score is better than the best one
        if val_dice_score > best_dice_score:
            best_dice_score = val_dice_score
            best_model_path = os.path.join(model_dir, f'best_model.pth')
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict()
                }, best_model_path)
            log_to_file(f"New Best Model Saved! Dice Score: {best_dice_score:.4f}", log_file)

    total_time = time.time() - total_start
    log_to_file(f"\nTotal Training Time: {total_time:.2f} seconds", log_file)


if __name__ == '__main__':
    main()