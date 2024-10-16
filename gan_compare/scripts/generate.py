import argparse
from dataclasses import asdict
from pathlib import Path
from time import time
import os
import logging
import cv2
import torch
from dacite import from_dict

from gan_compare.data_utils.utils import init_seed, interval_mapping
from gan_compare.training.gan_config import GANConfig
from gan_compare.training.io import load_yaml
from gan_compare.training.networks.generation.dcgan.dcgan_model import DCGANModel
from gan_compare.training.networks.generation.lsgan.lsgan_model import LSGANModel
from gan_compare.training.networks.generation.wgangp.wgangp_model import WGANGPModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_checkpoint_dir",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        default=None,
        help="Path to model checkpoint .pt file",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="How many samples to generate",
    )
    parser.add_argument(
        "--dont_show_images",
        action="store_true",
        help="Whether to show the generated images in UI.",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Whether to save the generated images.",
    )
    parser.add_argument(
        "--out_images_path",
        type=str,
        default=None,
        help="Directory to save the generated images in.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="The type of device on which images should be generated, e.g. cuda or cpu",
    )
    parser.add_argument(
        "--condition",
        type=int,
        default=None,
        help="Define the conditional input into the GAN i.e. a scalar between 0 and 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed used for random number generation i.e. random noise vector input into generator",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # initializing the random seed
    init_seed(args.seed)

    # Load model and config
    assert Path(
        args.model_checkpoint_dir
    ).is_dir(), f"The path to the model dir you provided does not point to a valid dir:{args.model_checkpoint_dir} "
    yaml_path = next(
        Path(args.model_checkpoint_dir).rglob("*.yaml")
    )  # i.e. "config.yaml")
    config_dict = load_yaml(path=yaml_path)
    # Needed to avoid WrongTypeError https://github.com/konradhalas/dacite/blob/master/dacite/core.py , Setting config={"check_types": False} not possible
    if config_dict["split_path"] is None: config_dict["split_path"] = ""

    config = from_dict(GANConfig, config_dict)
    print(asdict(config))
    logging.info("Loading model...")
    if config.gan_type == "dcgan":
        model = DCGANModel(
            config=config,
            dataloader=None,
            is_visualized=False,
        )
    elif config.gan_type == "lsgan":
        model = LSGANModel(
            config=config,
            dataloader=None,
            is_visualized=False,
        )
    elif config.gan_type == "wgangp":
        model = WGANGPModel(
            config=config,
            dataloader=None,
            is_visualized=False,
        )
    else:
        raise Exception(
            f"The gan_type ('{config.gan_type}') you provided via args is not valid."
        )
    condition = None
    if config.conditional or args.condition is not None:
        condition = f"{config.conditioned_on if config.conditioned_on is not None and config.conditioned_on in ['density', 'birads'] else f'{config.training_target}_{config.classes}'}"

    if config.conditional is False and args.condition is not None:
        print(
            f"You want to generate ROIs with condition ({condition}) = {args.condition}. Note that the GAN model you provided is not "
            f"conditioned on BIRADS. Therefore, it will generate unconditional random samples."
        )
        args.condition = None
    elif config.conditional is True and args.condition is not None:
        print(
            f"Conditional samples will be generate for condition {condition} = {args.condition}."
        )
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_checkpoint_path is None:
        try:
            args.model_checkpoint_path = next(
                Path(args.model_checkpoint_dir).rglob("*model.pt")
            )  # i.e. "model.pt"
        except StopIteration:
            try:
                # As there is no model.pt file, let's try to get the last item of the iterator instead, i.e. "300.pt"
                *_, args.model_checkpoint_path = Path(args.model_checkpoint_dir).rglob(
                    "*.pt"
                )
            except ValueError:
                pass

    # Let's validate the path to the model
    assert (
        args.model_checkpoint_path is not None
        and Path(args.model_checkpoint_path).is_file()
    ), f"There seems to be no model file with extension .pt stored in the model_checkpoint_dir you provided: {args.model_checkpoint_dir}. A model is expected to be in: {args.model_checkpoint_path}"

    # Save the images to model checkpoint folder
    if args.out_images_path is None and args.save_images:
        args.out_images_path = Path(args.model_checkpoint_dir / "samples")
    elif args.out_images_path is not None:
        args.out_images_path = Path(args.out_images_path)

    print(f"Generated samples will be stored in: {args.out_images_path}.")

    print(
        f"Now using model retrieved from: {args.model_checkpoint_path} to generate {args.num_samples} samples.."
    )

    img_list = model.generate(
        model_checkpoint_path=args.model_checkpoint_path,
        num_samples=args.num_samples,
        fixed_condition=args.condition,
        device=args.device,
    )

    # Show the images in interactive UI
    if args.dont_show_images is False:
        for img_ in img_list:
            img_ = interval_mapping(img_.transpose(1, 2, 0), 0.0, 1.0, 0, 255)
            img_ = img_.astype("uint8")
            cv2.imshow("sample", img_ * 2)
            k = cv2.waitKey()
            if k == 27 or k == 32:  # Esc key or space to stop
                break
        cv2.destroyAllWindows()

    if args.save_images:
        if not os.path.exists(args.out_images_path):
            # Create a new directory because it does not exist
            os.makedirs(args.out_images_path)
        for i, img_ in enumerate(img_list):
            if condition is not None:
                img_path = args.out_images_path / f"{config.gan_type}_{i}_{time()}_{condition}-{args.condition}.png"
            else:
                img_path = args.out_images_path / f"{config.gan_type}_{i}_{time()}.png"
            # print(min(img_))
            # print(max(img_))
            img_ = interval_mapping(img_.transpose(1, 2, 0), 0.0, 1.0, 0, 255)
            img_ = img_.astype("uint8")

            cv2.imwrite(str(img_path.resolve()), img_)
        print(f"Saved generated images to {args.out_images_path.resolve()}")
