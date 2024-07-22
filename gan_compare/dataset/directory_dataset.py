import os
import random
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision
from pathlib import Path
import logging

from gan_compare import constants
from gan_compare.dataset.base_dataset import BaseDataset
from gan_compare.training.base_config import BaseConfig


class DirectoryDataset(BaseDataset):
    """Directory based image dataset."""

    def __init__(
        self,
        subset:str,
        config: BaseConfig = None,
        transform: any = None,
        dataset_path: str = None,
        dataset_name: str = None,

    ):
        self.config = config
        self.subset = subset
        if dataset_name is not None:
            self.dataset_names = [dataset_name]
        else:
            self.dataset_names = config.data[self.subset].dataset_names
        self.paths = []
        if "train" in self.subset  and "no_training" in config.split_path:
            logging.info(f"Skipping adding images to {self.subset} DirectoryDataset, as split_path "
                         f"('{config.split_path}') indicates that only synthetic data should be used for "
                         f"training in this experiment.")
        else:
            for dataset_name in self.dataset_names:
                self.paths.extend([
                    os.path.join(dataset_path, dataset_name, self.subset, "is_benign_false", filename)
                    for filename in os.listdir(os.path.join(dataset_path, dataset_name, self.subset, "is_benign_false"))
                ])
                self.paths.extend([
                    os.path.join(dataset_path, dataset_name, self.subset, "is_benign_true", filename)
                    for filename in os.listdir(os.path.join(dataset_path, dataset_name, self.subset, "is_benign_true"))
                ])
        self.model_name = self.config.model_name
        self.transform = transform
        self.final_shape = (self.config.image_size, self.config.image_size) # assuming height==width

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int, to_save: bool = False, roi_type="mass"):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.paths[idx]
        assert ".png" in image_path, f"Expected .png in image_path, got {image_path}"
        # Already extracted images don't need cropping
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.config.model_name in constants.swin_transformer_names:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, self.final_shape, interpolation=cv2.INTER_AREA)
        if self.config.model_name not in constants.swin_transformer_names:
            sample = torchvision.transforms.functional.to_tensor(image[..., np.newaxis])
        else:
            sample = torchvision.transforms.functional.to_tensor(image)

        if self.transform:
            sample = self.transform(sample)

        # TODO make more intelligent labels once we have more complex GANs
        label = None
        if self.config.training_target == "biopsy_proven_status":
            # TODO watch out, this is an assumption. Checking if benign is part of filename
            if "is_benign_1" in Path(image_path).name:
                # ...is_benign_1.png = benign
                label = 1
            elif "is_benign_0" in Path(image_path).name:
                # ...is_benign_0.png = malignant
                label = 0
            else:
                raise Exception(f"Neither 'malignant' nor 'benign' in image path: {image_path}. Not sure which label to use in this case, please revise.")
        if label is None:
            logging.warning(f"label was {label} for training target {self.config.training_target} (image_path={image_path}). We reset label to 0 (e.g. not healthy). Please revise if this is in line with your intented experiment.")
            label = 0

        #return sample, label, image, metapoint.roi_type[0], metapoint.patch_id, mask
        logging.debug(f"{sample.shape}, {label}, {len(image)}, {roi_type}")
        return sample, label, image, roi_type, -1, #[]