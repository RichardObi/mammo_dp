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


class SyntheticDataset(BaseDataset):
    """Synthetic GAN-generated images dataset."""

    def __init__(
        self,
        config: BaseConfig = None,
        transform: any = None,
    ):
        self.paths = [
            os.path.join(config.synthetic_data_dir, filename)
            for filename in os.listdir(config.synthetic_data_dir)
        ]
        self.config = config
        self.model_name = self.config.model_name
        self.transform = transform
        self.final_shape = (self.config.image_size, self.config.image_size) # assuming height==width

    @staticmethod
    def _calculate_expected_length(current_length: int, shuffle_proportion: int) -> int:
        return int(shuffle_proportion / (1 - shuffle_proportion) * current_length)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int, to_save: bool = False, roi_type="mass"):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.paths[idx]
        assert ".png" in image_path
        # Synthetic images don't need cropping
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.config.model_name in constants.swin_transformer_names:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # mask = np.zeros(image.shape)
        # mask = mask.astype("uint8")

        # if to_save:
        #     # TODO decide whether we shouldn't be returning a warning here instead
        #     if self.conditional_birads:
        #         condition = f"{metapoint['birads'][0]}"
        #         return image, condition
        #     else:
        #         return image
        # scale
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
            if "benign" in Path(image_path).name:
                label = 1
            elif "malignant" in Path(image_path).name:
                label = 0
            else:
                raise Exception(f"Neither 'malignant' nor 'benign' in image path: {image_path}. Not sure which label to use in this case, please revise.")
        if label is None:
            logging.warning(f"label was {label} for training target {self.config.training_target} (image_path={image_path}). We reset label to 0 (e.g. not healthy). Please revise if this is in line with your intented experiment.")
            label = 0

        #return sample, label, image, metapoint.roi_type[0], metapoint.patch_id, mask
        logging.debug(f"{sample.shape}, {label}, {len(image)}, {roi_type}")
        return sample, label, image, roi_type, -1, #[]