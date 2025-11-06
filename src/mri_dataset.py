import os
import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import torch
import torch.utils.data as data  # Data handling utilities


import torchvision.transforms as tt  # Image transformations
import albumentations as A  # Image augmentations
ROOT_PATH_DEFAULT = '/data/mri_dataset'  # Default dataset root path

# Define transformations for training, validation, and testing datasets using Albumentations library.
train_transform = A.Compose([
    A.Resize(width=128, height=128, p=1.0),  # Resize images to 128x128 pixels
    A.HorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
    A.VerticalFlip(p=0.5),  # Apply vertical flip with 50% probability
    A.RandomRotate90(p=0.5),  # Rotate randomly by 90 degrees with 50% probability
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),  # Randomly shift, scale, and rotate
])

val_transform = A.Compose([
    A.Resize(width=128, height=128, p=1.0),  # Resize images to 128x128 pixels
    A.HorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability (for data augmentation)
])

test_transform = A.Compose([
    A.Resize(width=128, height=128, p=1.0),  # Resize images to 128x128 pixels
])


# Custom PyTorch Dataset class for loading images and masks from a DataFrame.
class BrainDataset(data.Dataset):
    def __init__(self, df, transform=None, normalize=True):
        self.df = df
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 0])
        image = np.array(image)/255.
        mask = cv2.imread(self.df.iloc[idx, 1], 0)
        mask = np.array(mask)/255.
        
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        
        image = image.transpose((2,0,1))
        image = torch.from_numpy(image).type(torch.float32)
        if self.normalize:
            image = tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
        mask = np.expand_dims(mask, axis=-1).transpose((2,0,1))
        mask = torch.from_numpy(mask).type(torch.float32)
        
        return image, mask

def build_mri_dataset(root_path: str = ROOT_PATH_DEFAULT,
                      val_size: float = 0.1,
                      test_size: float = 0.15,
                      random_state: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build a DataFrame of image/mask pairs and split into train/val/test.

    Args:
        root_path: root directory containing subdirectories with mask files
                   (expected mask filenames contain "_mask").
        val_size: proportion of the full dataset to reserve for validation.
        test_size: proportion of the remaining training portion to use as test
                   (applied after extracting validation set).
        random_state: RNG seed for reproducible splits.

    Returns:
        files_df, train_df, val_df, test_df (all pandas DataFrames)
    """
    pattern = os.path.join(root_path, '*', '*_mask*')
    mask_files = glob.glob(pattern)
    image_files = [m.replace('_mask', '') for m in mask_files]

    def diagnosis(mask_path: str) -> int:
        img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: unable to read mask file {mask_path}")
            return 0
        return 1 if np.max(img) > 0 else 0

    files_df = pd.DataFrame({
        "image_path": image_files,
        "mask_path": mask_files,
        "diagnosis": [diagnosis(x) for x in mask_files]
    })

    # Stratified split: first extract validation set from full dataset
    train_df, val_df = train_test_split(files_df, stratify=files_df['diagnosis'], test_size=val_size, random_state=random_state)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # From the remaining training portion, extract a test set (proportion of that set)
    train_df, test_df = train_test_split(train_df, stratify=train_df['diagnosis'], test_size=test_size, random_state=random_state)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return files_df, train_df, val_df, test_df


def get_mri_datasets(root_path: str = ROOT_PATH_DEFAULT,
                     val_size: float = 0.1,
                     test_size: float = 0.1,
                     random_state: int = 0,
                     train_transform_obj: Optional[A.BasicTransform] = None,
                     val_transform_obj: Optional[A.BasicTransform] = None,
                     test_transform_obj: Optional[A.BasicTransform] = None,
                     mem_train_transform_obj = None) -> Tuple[data.Dataset, data.Dataset, data.Dataset]:
    """
    Convenience wrapper that returns PyTorch Dataset objects for train/val/test.

    Args:
        root_path: dataset root path (same semantics as `build_mri_dataset`).
        val_size, test_size, random_state: split parameters passed to build_mri_dataset.
        train_transform_obj, val_transform_obj, test_transform_obj: optional
            Albumentations transform objects. If None, the module-level
            `train_transform`, `val_transform`, `test_transform` are used.

    Returns:
        (train_dataset, val_dataset, test_dataset) as instances of BrainDataset.
    """
    # Use default transforms if none provided
    if train_transform_obj is None:
        train_transform_obj = train_transform
    if val_transform_obj is None:
        val_transform_obj = val_transform
    if test_transform_obj is None:
        test_transform_obj = test_transform

    _, train_df, val_df, test_df = build_mri_dataset(root_path=root_path,
                                                    val_size=val_size,
                                                    test_size=test_size,
                                                    random_state=random_state)

    mem_dataset = BrainDataset(train_df, transform=mem_train_transform_obj, normalize=False)
    train_dataset = BrainDataset(train_df, transform=train_transform_obj)
    val_dataset = BrainDataset(val_df, transform=val_transform_obj)
    test_dataset = BrainDataset(test_df, transform=test_transform_obj)

    return mem_dataset, train_dataset, val_dataset, test_dataset