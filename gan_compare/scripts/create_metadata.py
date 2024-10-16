import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom as dicom
from tqdm import tqdm

from gan_compare.data_utils.utils import (
    generate_bcdr_metapoint,
    generate_cbis_ddsm_metapoints,
    generate_healthy_bcdr_metapoints,
    generate_healthy_inbreast_metapoints,
    generate_inbreast_metapoints,
    get_file_list,
    init_seed,
    load_inbreast_mask,
    read_csv,
    setup_logger,
    is_metapoint_duplicate,
)
from gan_compare.dataset.constants import (
    BCDR_HEALTHY_SUBDIRECTORIES,
    BCDR_SUBDIRECTORIES,
    CBIS_DDSM_CSV_DICT,
)
from gan_compare.dataset.metapoint import Metapoint
from gan_compare.paths import (
    BCDR_ROOT_PATH,
    CBIS_DDSM_ROOT_PATH,
    INBREAST_CSV_PATH,
    INBREAST_IMAGE_PATH,
    INBREAST_XML_PATH,
)

# python -m gan_compare.scripts.create_metadata --output_path setup/all_metadata_w_cbis_ddsm.json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default= ["bcdr"], #["cbis-ddsm"],#, "inbreast", "bcdr"],
        nargs="+",
        help="Name of the dataset of interest.",
    )
    parser.add_argument(
        "--bcdr_subfolder",
        type=str,
        nargs="+",
        #default=["d01", "d02"], # Outcommented on 16.06.2024
        default=["d01", "d02", "f01", "f02","f03"], #"dn01"
        help="Symbols of BCDR subdirectories to use.",
    )
    parser.add_argument(
        "--output_path",
        #required=True,
        default="all_metadata_bcdr_without_healthy.json",
        help="Path to json file to store metadata in."
    )
    parser.add_argument(
        "--per_image_count",
        type=int,
        default=5,
        help="Number of patches to generate from 1 healthy img.",
    )
    parser.add_argument(
        "--healthy_size",
        type=int,
        default=128,
        help="Size of patches to generate from 1 healthy img.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Seed the random generator. Can be used for example for generating different healthy patches.",
    )
    parser.add_argument(
        "--radiomics",
        action="store_true",
        help="Whether to compute radiomics features for each patch.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="gan_compare/configs/pretraining_experiments/classification_no_random_crop.yaml",
        help="Path to a yaml model config file used during training",
    )
    args = parser.parse_args()
    return args


def create_inbreast_metadata(
    patch_id: int,
    per_image_count: int = 5,
    target_size: int = 128,
    rng: np.random.Generator = np.random.default_rng(),
) -> Tuple[List[Metapoint], int]:
    metadata = []
    inbreast_df = read_csv(INBREAST_CSV_PATH)
    for filename in tqdm(get_file_list()):
        image_path = INBREAST_IMAGE_PATH / filename
        image_id, patient_id = image_path.stem.split("_")[:2]
        image_id = int(image_id)
        csv_metadata = inbreast_df.loc[inbreast_df["File Name"] == image_id].iloc[0]
        metapoints, patch_id = generate_healthy_inbreast_metapoints(
            patch_id=patch_id,
            image_id=image_id,
            patient_id=patient_id,
            csv_metadata=csv_metadata,
            image_path=image_path,
            per_image_count=per_image_count,
            size=target_size,
            bg_pixels_max_ratio=0.4,
            rng=rng,
        )
        metadata.extend(metapoints)
        ds = dicom.dcmread(image_path)
        xml_filepath = INBREAST_XML_PATH / f"{image_id}.xml"
        if xml_filepath.is_file():
            with open(xml_filepath, "rb") as patient_xml:
                # returns list of dictionaries, e.g. [{mask:mask1, roi_type:type1}, {mask:mask2, roi_type:type2}]
                mask_list = load_inbreast_mask(patient_xml, ds.pixel_array.shape)
        else:
            mask_list = [{"mask": np.zeros(ds.pixel_array.shape), "roi_type": ""}]
            print(f"No xml file found. Please review why. Path: {xml_filepath}")
            xml_filepath = ""
        for mask_dict in mask_list:
            lesion_metapoints, patch_id = generate_inbreast_metapoints(
                patch_id=patch_id,
                mask=mask_dict.get("mask"),
                image_id=image_id,
                patient_id=patient_id,
                csv_metadata=csv_metadata,
                image_path=image_path,
                roi_type=mask_dict.get("roi_type"),
            )
            # Add the metapoint objects of each contour to our metadata list
            metadata.extend(lesion_metapoints)
    return metadata, patch_id


def create_bcdr_metadata(
    patch_id: int,
    subdirectories_list: List[str],
    per_image_count: int = 5,
    target_size: int = 128,
    rng: np.random.Generator = np.random.default_rng(),
    filter_duplicates: bool = True,
) -> Tuple[List[Metapoint], int]:
    metadata = []
    if "dn01" in subdirectories_list:
        for subdirectory_symbol, subdirectory_name in BCDR_HEALTHY_SUBDIRECTORIES.items():
            csv_path = (
                BCDR_ROOT_PATH / subdirectory_name / f"bcdr_{subdirectory_symbol}_img.csv"
            )
            img_csv_df = pd.read_csv(csv_path)
            img_dir_path = BCDR_ROOT_PATH / subdirectory_name
            for _, row in tqdm(img_csv_df.iterrows(), total=img_csv_df.shape[0]):
                metapoints, patch_id = generate_healthy_bcdr_metapoints(
                    patch_id=patch_id,
                    image_dir_path=img_dir_path,
                    row_df=row,
                    per_image_count=per_image_count,
                    size=target_size,
                    bg_pixels_max_ratio=0.4,
                    rng=rng,
                )
                metadata.extend(metapoints)
    assert all(
        subdir in BCDR_SUBDIRECTORIES.keys() for subdir in subdirectories_list
    ), "Unknown subdirectory symbol"
    for subdirectory in subdirectories_list:
        csv_outlines_path = (
            BCDR_ROOT_PATH
            / BCDR_SUBDIRECTORIES[subdirectory]
            / f"bcdr_{subdirectory}_outlines.csv"
        )
        csv_outlines_path2 = (
            BCDR_ROOT_PATH
            / BCDR_SUBDIRECTORIES[subdirectory]
            / BCDR_SUBDIRECTORIES[subdirectory].replace("_dataset", "")
            / f"bcdr_{subdirectory}_outlines.csv"
        )
        try:
            outlines_df = pd.read_csv(csv_outlines_path)
        except:
            outlines_df = pd.read_csv(csv_outlines_path2)

        img_dir_path = BCDR_ROOT_PATH / BCDR_SUBDIRECTORIES[subdirectory]
        for _, row in outlines_df.iterrows():
            lesion_metapoint = generate_bcdr_metapoint(
                patch_id=patch_id, image_dir_path=img_dir_path, row_df=row
            )
            if filter_duplicates and is_metapoint_duplicate(lesion_metapoint=lesion_metapoint, metadata=metadata):
                continue # not adding this metapoint to avoid duplicates.
            patch_id += 1
            metadata.append(lesion_metapoint)
    return metadata, patch_id


def create_cbis_ddsm_metadata(
    patch_id: int,
    subsets: List[str],
    per_image_count: int = 5,
    target_size: int = 128,
    rng=np.random.default_rng(),
) -> Tuple[List[Metapoint], int]:

    metadata = []
    csv_paths = []
    # Select csv files with train/test and mass/calcification samples
    for subset in subsets:
        csv_paths.append(CBIS_DDSM_ROOT_PATH / CBIS_DDSM_CSV_DICT[subset + "_mass"])
        #csv_paths.append(CBIS_DDSM_ROOT_PATH / CBIS_DDSM_CSV_DICT[subset + "_calc"])

    for csv_path in csv_paths:
        cbis_ddsm_df = read_csv(csv_path, ",")
        print(cbis_ddsm_df.head())

        # Corrections for consistency in metadata.json file
        cbis_ddsm_df = cbis_ddsm_df.replace("MALIGNANT", "malignant")
        cbis_ddsm_df = cbis_ddsm_df.replace("BENIGN", "benign")
        cbis_ddsm_df = cbis_ddsm_df.replace(
            "BENIGN_WITHOUT_CALLBACK", "benign_without_callback"
        )
        cbis_ddsm_df = cbis_ddsm_df.replace("RIGHT", "R")
        cbis_ddsm_df = cbis_ddsm_df.replace("LEFT", "L")
        cbis_ddsm_df = cbis_ddsm_df.replace(r"\n", "", regex=True)
        cbis_ddsm_df = cbis_ddsm_df.rename(columns={"breast_density": "breast density"})
        cbis_ddsm_df = cbis_ddsm_df.rename(columns={"mass_shape": "mass shape"})
        cbis_ddsm_df = cbis_ddsm_df.rename(columns={"mass_margins": "mass margins"})
        cbis_ddsm_df["breast density"] = cbis_ddsm_df["breast density"].replace(0, -1)

        for _, sample in tqdm(cbis_ddsm_df.iterrows(), total=cbis_ddsm_df.shape[0]):
            image_path = CBIS_DDSM_ROOT_PATH / sample["image file path"].replace(
                "000000.dcm", "1-1.dcm"
            )
            image_id = int(image_path.parent.suffix.replace(".", ""))
            if image_path.is_file():
                ds = dicom.dcmread(image_path)
                mask_path = None
                mask_directory = (
                    CBIS_DDSM_ROOT_PATH / Path(sample["ROI mask file path"]).parent
                )
                for filename in os.listdir(mask_directory):
                    file_path = mask_directory / filename
                    ds_mask = dicom.dcmread(file_path)
                    # Patches and masks can have the same name, so we need to distinguish them by size threshold
                    if ds_mask.pixel_array.shape[0] > ds.pixel_array.shape[0] * 0.5:
                        if ds_mask.pixel_array.shape == ds.pixel_array.shape:
                            mask_path = file_path
                            mask = ds_mask.pixel_array
                        else:
                            # Mask size mismatch. Rescaling applied.
                            mask = cv2.resize(
                                ds_mask.pixel_array,
                                list(ds.pixel_array.shape)[::-1],
                                interpolation=cv2.INTER_NEAREST,
                            )
                        mask_path = file_path
                if not mask_path:
                    mask = np.zeros(ds.pixel_array.shape)
                    print(f"No mask found. Please review why. Path: {mask_directory}")

                lesion_metapoints, patch_id = generate_cbis_ddsm_metapoints(
                    patch_id=patch_id,
                    mask=mask,
                    image_id=image_id,
                    patient_id=sample["patient_id"],
                    csv_metadata=sample,
                    image_path=image_path,
                    roi_type=sample["abnormality type"],
                )

                # Add the metapoint objects of each contour to our metadata list
                metadata.extend(lesion_metapoints)
    return metadata, patch_id


def generate_additional_attributes_for_metadata(metadata, args):
    if args.radiomics:
        from gan_compare.data_utils.radiomics_generation import generate_radiomics
        metadata = generate_radiomics(metadata, args.config_path)

    return metadata


if __name__ == "__main__":
    _ = setup_logger()
    args = parse_args()
    init_seed(args.seed)  # setting the seed from the args
    rng = np.random.default_rng(
        args.seed
    )  # seed the random generator. TODO: use np.random.randint() instead of rng.integers()
    metadata = []
    patch_id = 0
    if "cbis-ddsm" in args.dataset:
        print("Generating metadata for CBIS-DDSM")
        cbis_ddsm_metadata, patch_id = create_cbis_ddsm_metadata(
            patch_id=patch_id,
            subsets=["train", "test"],  # FIXME Only test data?
            per_image_count=args.per_image_count,
            target_size=args.healthy_size,
            rng=rng,
        )
        metadata.extend(cbis_ddsm_metadata)

    if "bcdr" in args.dataset:
        bcdr_metadata, patch_id = create_bcdr_metadata(
            patch_id=patch_id,
            subdirectories_list=args.bcdr_subfolder,
            per_image_count=args.per_image_count,
            target_size=args.healthy_size,
            rng=rng,
        )
        metadata.extend(bcdr_metadata)

    if "inbreast" in args.dataset:
        inbreast_metadata, patch_id = create_inbreast_metadata(
            patch_id=patch_id,
            per_image_count=args.per_image_count,
            target_size=args.healthy_size,
            rng=rng,
        )
        metadata.extend(inbreast_metadata)

    metadata = generate_additional_attributes_for_metadata(metadata, args)

    # Output metadata as json file to specified location on disk
    outpath = Path(args.output_path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    metadata_list_of_dict = [dataclasses.asdict(metapoint) for metapoint in metadata]
    with open(outpath.resolve(), "w") as outfile:
        json.dump(metadata_list_of_dict, outfile, indent=4)
    print(f"Saved {len(metadata_list_of_dict)} metapoints to {outpath.resolve()}")
