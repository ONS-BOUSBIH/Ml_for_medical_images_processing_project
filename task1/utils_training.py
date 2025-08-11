import torch
from monai.metrics import DiceMetric
from tqdm import tqdm
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from datetime import datetime
import time

# =====================================
#  Utils file for training
# =====================================


def log_to_file(message, log_file):
    """
    Write a message to the specified log file.
    
    Args:
        message (str): The message to be logged.
        log_file (str): Path to the log file.
    """
    with open(log_file, "a") as f:
        f.write(message + "\n")


def log_transforms(transforms, log_file, phase):
    """
    Log the preprocessing transforms applied to the dataset.
    
    Args:
        transforms (Compose): A composition of data transformation operations.
        log_file (str): Path to the log file.
        phase (str): training or validation
    """
    log_to_file(f"\n{phase} Preprocessing Pipeline:", log_file)
    for t in transforms.transforms:
        log_to_file(f" - {t.__class__.__name__}: {t}", log_file)


def train(train_loader, model, optimizer, loss_fn, device, log_file):
    """
    Run the Training loop.
    
    Args:
        train_loader (DataLoader): Dataloader for training dataset.
        model (torch.nn.Module): Model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        loss_fn (callable): Loss function to compute training loss.
        device (torch.device): Device to run training on.
        log_file (str): Log file path.

    Returns:
        tuple: Average loss and total training time for the epoch.
    """
    model.train()
    total_loss = 0
    start_time = time.time()

    for img, mask in tqdm(train_loader, desc="Training", unit="batch"):
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        optimizer.zero_grad()
        loss = loss_fn(pred, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    # Write the average loss for every epoch in the log file
    log_to_file(f"Training Loss: {avg_loss:.4f} | Time: {epoch_time:.2f} seconds", log_file)
    return avg_loss, epoch_time


def validate(val_loader, model, loss_fn, device, log_file):
    """
    Run the validation loop.
    
    Args:
        val_loader (DataLoader): Dataloader for validation dataset.
        model (torch.nn.Module): Trained model to validate.
        loss_fn (callable): Loss function used for evaluation.
        device (torch.device): Device to run validation on.
        log_file (str): Log file path.

    Returns:
        tuple: Average loss, average Dice score, and total validation time.
    """
    model.eval()
    total_loss = 0
    # Initialize dice metric for evaluation
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    # Define post-processing transforms for predictions and labels, converting them to one-hot encoding
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    start_time = time.time()
    with torch.no_grad():
        for img, mask in tqdm(val_loader, desc="Validation", unit="batch"):
            img, mask = img.to(device), mask.to(device)
            # Using sliding window to feed the image to the model as 4 overlapping volumes
            pred = sliding_window_inference(img, (96, 96, 96),4, model)
            loss = loss_fn(pred, mask)
            total_loss += loss.item()

            preds_list = [pred[i] for i in range(pred.shape[0])]
            pred = [post_pred(m) for m in preds_list]
            dice = dice_metric(pred, mask)

        avg_dice_score = dice_metric.aggregate().item()
        dice_metric.reset()
        avg_loss = total_loss / len(val_loader)

    epoch_time = time.time() - start_time
    # Log validation result for each epoch
    log_to_file(
        f"Validation Loss: {avg_loss:.4f}, Dice Score: {avg_dice_score:.4f} | Time: {epoch_time:.2f} seconds",
        log_file,
    )
    return avg_loss, avg_dice_score, epoch_time
