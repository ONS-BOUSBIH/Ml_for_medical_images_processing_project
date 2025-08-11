import torch
import os
import logging
import argparse
import nibabel as nib
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from monai.networks.nets import UNETR
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    RemoveSmallObjects,
    KeepLargestConnectedComponent,
)
from utils_data_loading import NiftiDataset, resample_image, validation_transforms

# =====================================
#  File for running infernece
# =====================================


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def inference_compute_mean_dice_with_pp(inference_loader, device, model, resampling, post_processing_param, postprocessing_method):
    """Runs inference and computes the mean Dice score on a test dataset."""
    
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)
    dice_scores = []

    with torch.no_grad():
        for img, mask in tqdm(inference_loader, desc="Testing", unit="batch"):
            try:
                img, mask = img[0].to(device), mask[0].to(device)

                # Run inference
                pred = sliding_window_inference(img, (96, 96, 96), 4, model)
                pred = torch.argmax(pred, dim=1, keepdim=False)

                # Apply post-processing method
                if postprocessing_method == "remove_small":
                    pred = RemoveSmallObjects(min_size=post_processing_param)(pred)
                elif postprocessing_method == "keep_largest":
                    pred = KeepLargestConnectedComponent(is_onehot=False, connectivity=3, num_components=post_processing_param)(pred)
                else:
                    logging.warning(f"Unknown post-processing method: {postprocessing_method}. Using 'remove_small' as default.")
                    pred = RemoveSmallObjects(min_size=min_object_size)(pred)

                # Convert to one-hot encoding and permute
                pred = torch.nn.functional.one_hot(pred, num_classes=2).permute(0, 4, 1, 2, 3)

                # Resample prediction to match mask shape
                pred = resampling(pred, mask.shape[-3:])

                # Compute dice score
                pred_list = [post_pred(p) for p in pred]
                mask_list = [post_label(m) for m in mask]
                dice = dice_metric(pred_list, mask_list)
                dice_scores.extend(dice.cpu().tolist())

                last_pred = torch.argmax(pred[0], dim=0)
                last_mask = mask

            except Exception as e:
                logging.error(f"Error during inference: {e}")

        avg_dice_score = dice_metric.aggregate().item()
        dice_metric.reset()

    logging.info(f"Average Dice Score: {avg_dice_score}")
    return avg_dice_score, last_pred, last_mask, dice_scores


def main(args):
    """Main function to load the model, prepare the dataset, and run inference."""
    
    # Choose GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Define paths
    model_dir = args.model_dir
    ckpt_file = "best_model.pth"
    ckpt_path = os.path.join(model_dir, ckpt_file)
    data_dir_path = args.data_dir

    # Load model
    logging.info("Loading UNETR model...")
    model = UNETR(
        in_channels=1,
        out_channels=2,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)

    # Load model checkpoint (weights only)
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info("Model checkpoint loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model checkpoint: {e}")
        return

    # Prepare dataset
    logging.info("Preparing dataset...")

    # Get image and mask paths
    image_paths, mask_paths = [], []
    for f in sorted(os.listdir(data_dir_path)):
        if "img" in f:
            idx = f.split('.')[0]
            mask_path = os.path.join(data_dir_path, f"{idx}.label.nii.gz")
            if os.path.isfile(mask_path):
                image_paths.append(os.path.join(data_dir_path, f))
                mask_paths.append(mask_path)

    # Split dataset
    train_split_idx = int(args.train_split_rate * len(image_paths))
    val_split_idx = int(args.val_split_rate * len(image_paths)) + train_split_idx
    test_paths = [image_paths[val_split_idx:], mask_paths[val_split_idx:]]

    # Create dataset and DataLoader
    inference_dataset = NiftiDataset(test_paths[0], test_paths[1], validation_transforms())
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Run inference
    inference_compute_mean_dice_with_pp(
        inference_loader=inference_loader,
        device=device,
        model=model,
        resampling=resample_image,
        post_processing_param=args.post_processing_param,
        postprocessing_method=args.postprocessing_method
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UNETR model inference and compute Dice scores.")

    # Add arguments
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./unetr_test",
        help="Path to the directory containing the trained model."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/training_data/",
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--post_processing_param",
        type=int,
        default=64,
        help="For 'remove_small', this is the minimum object size to retain. For 'keep_largest', this is the number of largest components to retain."
    )
    parser.add_argument(
        "--train_split_rate",
        type=float,
        default=0.7,
        help="Ratio of training split."
    )
    parser.add_argument(
        "--val_split_rate",
        type=float,
        default=0.25,
        help="Ratio of validation split."
    )
    parser.add_argument(
        "--postprocessing_method",
        type=str,
        choices=["remove_small", "keep_largest"],
        default="remove_small",
        help="Choose the post-processing method: 'remove_small' (default) or 'keep_largest'."
    )

    args = parser.parse_args()
    main(args)