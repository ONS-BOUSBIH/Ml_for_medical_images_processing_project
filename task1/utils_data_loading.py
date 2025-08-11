import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from monai.data import Dataset
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    Lambda,
    SpatialPad,
)

# =====================================
#  Utils file for data loading
# =====================================

# =====================================
#  Dataset Classes
# =====================================

class NiftiDataset(Dataset):
    """
    Custom dataset class for loading NIfTI images and masks.

    Args:
        image_paths (list): List of paths to NIfTI images.
        mask_paths (list): List of paths to the corresponding masks.
        transforms (callable, optional): Transformations to apply to the images and masks.
    """

    def __init__(self, image_paths, mask_paths, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        sample = {"image": self.image_paths[idx], "label": self.mask_paths[idx]}
        if self.transforms:
            sample = self.transforms(sample)

        if isinstance(sample, list):
            images = [s["image"].clone().detach() for s in sample]
            labels = [s["label"].clone().detach() for s in sample]
        else:
            images = [sample["image"]]
            labels = [sample["label"]]

        return images, labels


class CustomNiftiDatasetSubmission(Dataset):
    """
    Dataset class for handling NIfTI images for inference (submission).
    
    Args:
        image_paths (list): List of paths to images.
        transforms (callable): Transformations to apply to the images.
    """

    def __init__(self, image_paths, transforms):
        data = []
        for img_path in image_paths:
            img_nifti = nib.load(img_path)
            original_shape = torch.Size(img_nifti.shape)  # Get original image shape
            data.append({
                "image": img_path,
                "filename": img_path,
                "original_shape": original_shape
            })

        super().__init__(data, transform=transforms)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        return item["image"], item["filename"], item["original_shape"]


# =====================================
#  Data Transformations
# =====================================

def training_transforms():
    """
    Define the set of transformations applied during training.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-750, a_max=500, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4, allow_smaller=True
        ),
        Lambda(lambda x: {
            "image": SpatialPad(spatial_size=(96, 96, 96))(x["image"]),
            "label": SpatialPad(spatial_size=(96, 96, 96))(x["label"])
        }),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.1),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.1),
        RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.1),
        RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ])


def validation_transforms():
    """
    Define the set of transformations applied during validation.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-750, a_max=500, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ])


def submission_transforms():
    """
    Define the set of transformations applied during inference.
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
    ])


# =====================================
#  Image Resampling
# =====================================

def resample_image(image, target_shape):
    """
    Resample an image tensor to a given shape.

    Args:
        image (torch.Tensor): Input image tensor.
        target_shape (tuple): Target shape for resampling.

    Returns:
        torch.Tensor: Resampled image tensor.
    """
    target_shape = [int(dim) for dim in target_shape]
    image = image.float()
    return F.interpolate(image, size=target_shape, mode='trilinear', align_corners=False)


# =====================================
#  Data Collation Function
# =====================================

def collate_multiple_crops(batch):
    """
    Custom collate function to handle multiple crops in a batch.

    Args:
        batch (list): A batch of images and labels.

    Returns:
        tuple: Stacked images and labels as tensors.
    """
    images, labels = [], []
    for image_list, label_list in batch:
        images.extend(image_list)  # Flatten list of crops
        labels.extend(label_list)

    return torch.stack(images), torch.stack(labels)