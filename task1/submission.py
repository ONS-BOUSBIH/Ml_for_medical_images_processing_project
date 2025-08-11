import torch
import os
import logging
import argparse
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from monai.networks.nets import UNETR
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    RemoveSmallObjects,
    KeepLargestConnectedComponent,
)
from utils_data_loading import CustomNiftiDatasetSubmission, resample_image, submission_transforms

# =====================================
#  File for creating the submission data
# =====================================

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def compute_and_save_submission(inference_loader, device, model, resampling, post_processing_param, postprocessing_method, save_dir):
    """Runs inference on a dataset and saves predictions as NIfTI files."""
    
    model.eval()
    post_pred = AsDiscrete(argmax=True, to_onehot=2)

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for img, filename, original_shape in inference_loader:
            try:
                img = img.to(device)

                # Extract filename
                image_id = os.path.basename(filename[0]).split('.')[0]

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
                    pred = RemoveSmallObjects(min_size=post_processing_param)(pred)

                # Convert to one-hot encoding and permute to match expected shape
                pred = torch.nn.functional.one_hot(pred, num_classes=2).permute(0, 4, 1, 2, 3)

                # Resample prediction to match original image shape
                pred = resampling(pred, original_shape)

                # Convert predictions to correct format
                pred = post_pred(pred[0])
                last_pred = torch.argmax(pred, dim=0).cpu().numpy()

                # Save as a NIfTI file
                nifti_img = nib.Nifti1Image(last_pred.astype(np.uint8), np.eye(4))  # Adjust affine if needed
                save_path = os.path.join(save_dir, f"{image_id}.label.nii.gz")
                nib.save(nifti_img, save_path)

                logging.info(f"Saved prediction: {save_path}")

            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")

    logging.info("Inference and saving completed.")


def main(args):
    """Main function to load model, prepare dataset, and run inference."""
    
    # Choose GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Define paths
    model_dir = args.model_dir
    ckpt_file = "best_model.pth"
    ckpt_path = os.path.join(model_dir, ckpt_file)
    data_dir_path = args.data_dir
    save_directory = args.save_dir

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

    # Load model checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info("Model checkpoint loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model checkpoint: {e}")
        return

    # Prepare dataset
    logging.info("Preparing dataset...")

    # Get image paths
    image_paths = [os.path.join(data_dir_path, f) for f in sorted(os.listdir(data_dir_path)) if 'img' in f]

    if not image_paths:
        logging.error("No images found in the dataset directory!")
        return

    # Create dataset and dataloader
    dataset = CustomNiftiDatasetSubmission(image_paths, submission_transforms())
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Run inference and save predictions
    logging.info(f"Running inference with post-processing method: {args.postprocessing_method} and parameter: {args.post_processing_param}")
    compute_and_save_submission(
        inference_loader=data_loader,
        device=device,
        model=model,
        resampling=resample_image,
        post_processing_param=args.post_processing_param,
        postprocessing_method=args.postprocessing_method,
        save_dir=save_directory,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UNETR model inference on medical images.")
    
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
        default="/data/test_set/", 
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="./submission/acorn/test_set", 
        help="Directory where predictions will be saved."
    )
    parser.add_argument(
        "--post_processing_param", 
        type=int, 
        default=64, 
        help="For 'remove_small', this is the minimum object size to retain. For 'keep_largest', this is the number of largest components to retain."
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