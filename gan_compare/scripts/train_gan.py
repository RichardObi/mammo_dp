from __future__ import print_function

import argparse
import logging
import os
from dataclasses import asdict
from pathlib import Path

import cv2
import torch
import torchvision.transforms as transforms
from dacite import from_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from gan_compare.data_utils.utils import collate_fn, init_seed, setup_logger
from gan_compare.dataset.mammo_dataset import MammographyDataset
from gan_compare.training.gan_config import GANConfig
from gan_compare.training.io import load_yaml
from gan_compare.training.networks.generation.dcgan.dcgan_model import DCGANModel
from gan_compare.training.networks.generation.lsgan.lsgan_model import LSGANModel
from gan_compare.training.networks.generation.wgangp.wgangp_model import WGANGPModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="gan_compare/configs/gan/dcgan_config.yaml",
        help="Path to a yaml model config file",
    )
    parser.add_argument(
        "--save_dataset",
        action="store_true",
        help="Whether to save the dataset samples.",
    )
    parser.add_argument(
        "--out_dataset_path",
        type=str,
        default="../../data/mammo_dbr/dataset16062024/",
        help="Directory to save the dataset samples in.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--use_dp",
        action="store_true",
        help="The switch that decides whether DP should be enabled or not during training.",
    )
    parser.set_defaults(use_dp=False)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to be used for training/testing",
    )
    args = parser.parse_args()
    return args


def train_gan(args):

    # Parse config file
    config_dict = load_yaml(path=args.config_path)
    config = from_dict(GANConfig, config_dict)
    logfilename, logfile_path = setup_logger(logfile_path=config.logfile_path, log_level=config.log_level)

    logging.info(f"GAN config dict: {asdict(config)}")
    logging.info(f"GAN args dict: {args}")

    init_seed(args.seed)  # initializing the random seed

    transform_to_use = None
    if config.is_training_data_augmented:
        transform_to_use = transforms.Compose(
            [
                #transforms.Normalize((0.), (1.)), # between -1 and 1 https://github.com/soumith/ganhacks #1
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # scale: min 0.75 of original image pixels should be in crop, radio: randomly between 3:4 and 4:5
                #transforms.RandomResizedCrop(size=config.image_size, scale=(0.75, 1.0), ratio=(0.75, 1.3333333333333333)),
                transforms.RandomResizedCrop(size=config.image_size, scale=(0.95, 1.0), ratio=(0.95, 1.1)),
                # RandomAffine is not used to avoid edges with filled pixel values to avoid that the generator learns this bias
                # which is not present in the original images.
                # transforms.RandomAffine(),
            ]
        )

    subset = "test" #"val" #"train" #"test"
    dataset = MammographyDataset(
        metadata_path=config.metadata_path,
        split_path=config.split_path,
        subset=subset,
        config=config,
        sampling_ratio=config.train_sampling_ratio,
        transform=transform_to_use,
    )

    logging.info(
        f"Loaded dataset {dataset.__class__.__name__}, with augmentations(?): {config.is_training_data_augmented}"
    )

    if args.save_dataset:
        output_dataset_dir = Path(args.out_dataset_path)
        if not output_dataset_dir.exists():
            os.makedirs(output_dataset_dir.resolve())
        for i in tqdm(range(len(dataset))):
            # print(dataset[i])
            # Plot some training images
            items = dataset.__getitem__(i, return_metapoint=True)
            if items is None:
                continue
            try:
                (
                    sample,
                    condition,
                    image,
                    roi_type,
                    patch_id,
                    metapoint
                ) = items
            except:
                (
                sample,
                condition,
                image,
                roi_type,
                patch_id,
                ) = items
            out_image_path = (
                f"patient{metapoint.patient_id}_image{metapoint.image_id}_{roi_type}{patch_id}_{config.classes}_{condition}.png"
                if config.conditional
                else f"patient{metapoint.patient_id}_image{metapoint.image_id}_{roi_type}{patch_id}.png"
            )
            if config.conditional and condition==1:
                output_dataset_dir_ = output_dataset_dir / metapoint.dataset / subset /  f"{config.classes}_true"
            elif config.conditional and condition==0:
                output_dataset_dir_ = output_dataset_dir / metapoint.dataset / subset / f"{config.classes}_false"
            os.makedirs(output_dataset_dir_.resolve(), exist_ok=True)
            cv2.imwrite(str(output_dataset_dir_ / out_image_path), image)
        logging.info(f"Saved dataset samples to {output_dataset_dir.resolve()}")
        exit()

    # drop_last is true to avoid batch_size of 1 that throws an Value Error in BatchNorm.
    # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/5
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        num_workers=config.workers,
        batch_size=config.batch_size,
        collate_fn=collate_fn,  # Filter out None returned by DataSet.
        drop_last=True,
    )

    # Emptying the cache for GPU RAM i.e. to avoid cuda out of memory issues
    torch.cuda.empty_cache()
    logging.info("Loading model...")
    if config.gan_type == "dcgan":
        model = DCGANModel(
            config=config,
            dataloader=dataloader,
            use_dp=args.use_dp,
            device=args.device,
        )
    elif config.gan_type == "lsgan":
        model = LSGANModel(
            config=config,
            dataloader=dataloader,
            use_dp=args.use_dp
        )
    elif config.gan_type == "wgangp":
        model = WGANGPModel(
            config=config,
            dataloader=dataloader,
            use_dp=args.use_dp,
        )
    else:
        raise Exception(
            f"The gan_type ('{config.gan_type}') you provided via args is not valid."
        )

    logging.info(f"Initialized {config.gan_type} model. Now starting training...")
    model.train()


if __name__ == "__main__":
    args = parse_args()
    train_gan(args)
