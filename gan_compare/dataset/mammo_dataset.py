import logging
from typing import Optional
import random

import cv2
import numpy as np
import torch
import torchvision

from gan_compare.data_utils.utils import (
    get_image_from_metapoint,
    get_mask_from_metapoint,
)
from gan_compare.dataset.base_dataset import BaseDataset
from gan_compare.dataset.metapoint import Metapoint
from gan_compare.training.base_config import BaseConfig
from gan_compare.training.io import load_json
from gan_compare import constants


class MammographyDataset(BaseDataset):
    """Mammography dataset class."""

    def __init__(
        self,
        metadata_path: str,
        config: BaseConfig,
        split_path: Optional[str] = None,
        subset: str = "train",
        crop: bool = True,
        min_size: int = 128,
        margin: int = 60,
        conditional_birads: bool = False,
        # Setting this to True will result in BiRADS annotation with 4a, 4b, 4c split to separate classes
        transform: any = None,
        sampling_ratio: float = 1.0,
        filter_metadata: bool = True,
        density_filter:list = None,
        mass_margin_filter:list = None,
        mass_subtlety_filter:list = None,
        return_indices:bool = False
    ):
        super().__init__(
            metadata_path=metadata_path,
            crop=crop,
            min_size=min_size,
            margin=margin,
            conditional_birads=conditional_birads,
            transform=transform,
            config=config,
            subset=subset,
        )

        if not split_path and self.config.split_path:
            split_path = self.config.split_path

        if hasattr(self.config, 'binary_classification') and self.config.binary_classification:

            assert split_path is not None, "Missing split path!"
        # We may also want to split the dataset if not using binary classification i.e. in GAN training.
        logging.info(
            f"Number of {subset} metadata before filtering patients: {len(self.metadata_unfiltered)}"
        )
        if split_path is not None and split_path != 'None' and filter_metadata:

            split_dict = load_json(split_path)
            #print(f"split_dict.keys(): {split_dict.keys()}")
            self.patient_ids = split_dict[subset]
            self.metadata.extend(
                [
                    metapoint
                    for metapoint in self.metadata_unfiltered
                    if metapoint.patient_id in self.patient_ids
                ]
            )
            logging.info(
                f"Number of {subset} metadata after filtering patients: {len(self.metadata)}"
            )
        else:
            logging.info(
                f"No filtering based on train, val, test: split_path was {split_path}: All metapoints from '{metadata_path}' from datasets {self.config.data} will be returned by dataset."
            )
            self.metadata.extend(self.metadata_unfiltered)


        if filter_metadata:
            # filter datasets of interest
            self.metadata = [
                metapoint
                for metapoint in self.metadata
                if metapoint.dataset in self.config.data[subset].dataset_names
            ]
            logging.info(
                f"Number of {subset} metadata after filtering dataset_names (allowed datasets: {self.config.data[subset].dataset_names}): {len(self.metadata)}"
            )
            #logging.info(f"metadata: {self.metadata}")

            # filter roi types of interest
            self.metadata = [
                metapoint
                for metapoint in self.metadata
                if any(
                    roi_type in self.config.data[subset].roi_types
                    for roi_type in metapoint.roi_type
                )
            ]
            logging.info(
                f"Number of {subset} metadata after filtering roi_types (allowed roi_types: {self.config.data[subset].roi_types}): {len(self.metadata)}"
            )
            if density_filter is not None:
                self.metadata = [
                    metapoint
                    for metapoint in self.metadata
                    if metapoint.density in density_filter
                ]
                logging.info(
                    f"Number of {subset} metadata after filtering density (allowed densities: {density_filter}): {len(self.metadata)}"
                )
            if mass_margin_filter is not None:
                if "!" in mass_margin_filter:
                    self.metadata = [
                        metapoint
                        for metapoint in self.metadata
                        if not any(mass_margin.lower() in metapoint.mass_margins.lower() for mass_margin in mass_margin_filter)
                    ]
                else:
                    self.metadata = [
                        metapoint
                        for metapoint in self.metadata
                        if any(mass_margin.lower() in metapoint.mass_margins.lower() for mass_margin in mass_margin_filter)
                    ]
                mass_margin_filter = set([metapoint.mass_margins for metapoint in self.metadata])  # Caution: Expensive!
                logging.info(f"Number of {subset} metadata after filtering mass margins (allowed margin types: {mass_margin_filter}): {len(self.metadata)}")

            if mass_subtlety_filter is not None:
                self.metadata = [
                    metapoint
                    for metapoint in self.metadata
                    if metapoint.subtlety in mass_subtlety_filter
                ]
                logging.info(
                    f"Number of {subset} metadata after filtering mass subtlety (allowed subtleties: {mass_subtlety_filter}): {len(self.metadata)}"
                )
            # filter labels that are unknown i.e. -1 for is_benign CLF
            if self.config.classes == "is_benign":
                self.metadata = [
                    metapoint
                    for metapoint in self.metadata
                    if metapoint.is_benign != -1
                ]
                logging.info(
                    f"Number of {subset} metadata after filtering classes where 'is_benign' is unknown: {len(self.metadata)}"
                )
                logging.debug(f"metadata after filtering classes where 'is_benign' is unknown is tested via: metapoint.is_benign != -1)")

            # filter labels that are unknown i.e. -1 for is_healthy CLF
            elif self.config.classes == "is_healthy":
                self.metadata = [
                    metapoint
                    for metapoint in self.metadata
                    if metapoint.is_healthy != -1
                ]
                logging.info(
                    f"Number of {subset} metadata after filtering classes where 'is_healthy' is unknwon (metapoint.is_healthy != -1): {len(self.metadata)}"
                )

        if sampling_ratio < 1.0:  # don't shuffle if we want to keep all the data
            random.seed(config.seed)
            self.metadata = random.sample(
                self.metadata,
                int(sampling_ratio * len(self.metadata)),
            )
            logging.info(
                f"Number of {subset} metadata after random sampling (with sampling_ratio={sampling_ratio}, and random seed={config.seed}): {len(self.metadata_unfiltered)}"
            )

        logging.info(f"Appended metadata. Final Metadata size: {len(self.metadata)}")

        if config.is_regression and self.normalize_output:
            self.normalize_output_data(self.metadata)
        self.return_indices = return_indices

    def __getitem__(self, idx: int, return_metapoint: bool = False):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metapoint = self.metadata[idx]
        assert isinstance(metapoint, Metapoint)

        image = get_image_from_metapoint(metapoint)
        mask = get_mask_from_metapoint(metapoint)

        margin = 0 if metapoint.is_healthy else self.margin
        x, y, w, h = self.get_crops_around_bbox(
            metapoint.bbox,
            margin=margin,
            min_size=self.min_size,
            image_shape=image.shape,
            config=self.config,
        )

        image, mask = image[y : y + h, x : x + w], mask[y : y + h, x : x + w]

        if not np.any(mask) and not metapoint.is_healthy:
            logging.debug(f"No mask found for {metapoint.patient_id}")

        if self.config.model_name in constants.swin_transformer_names:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # scale
        image = cv2.resize(image, self.final_shape, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.final_shape, interpolation=cv2.INTER_AREA)
        if self.config.model_name not in constants.swin_transformer_names:
            sample = torchvision.transforms.functional.to_tensor(image[..., np.newaxis])
        else:
            sample = torchvision.transforms.functional.to_tensor(image)

        if self.transform:
            sample = self.transform(sample)

        label = (
            self.retrieve_condition(metapoint)
            if (self.config.conditional and self.config.conditioned_on is not None and self.config.conditioned_on in ["density", "birads"] ) # we may want to condition on the label
            else self.determine_label(metapoint)
        )
        logging.debug(f"{sample.shape}, {label}, {len(image)}, {metapoint.roi_type[0]}, {metapoint.patch_id}")
        if self.return_indices:
            if return_metapoint:
                return sample, label, image, metapoint.roi_type[0], metapoint.patch_id, idx, metapoint, #mask
            return sample, label, image, metapoint.roi_type[0], metapoint.patch_id, idx  # mask
        else:
            if return_metapoint:
                return sample, label, image, metapoint.roi_type[0], metapoint.patch_id, metapoint, #mask
            return sample, label, image, metapoint.roi_type[0], metapoint.patch_id, #mask

