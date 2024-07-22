import dataclasses
import json
import logging
from pathlib import Path
from typing import List

import SimpleITK as sitk
from dacite import from_dict
from radiomics import featureextractor
from tqdm import tqdm

from gan_compare.dataset.mammo_dataset import MammographyDataset
from gan_compare.dataset.metapoint import Metapoint
from gan_compare.training.classifier_config import ClassifierConfig
from gan_compare.training.io import load_yaml


def generate_radiomics(
    metadata: List[Metapoint],
    config_path: str,
    features_to_compute: List[str] = [
        "firstorder",
        "glcm",
        "glrlm",
        "gldm",
        "glszm",
        "ngtdm",
    ],
) -> List[Metapoint]:
    """
    Generate radiomics features for each metapoint
    :param metadata: list of metapoints
    :param features_to_compute: list of radiomics features to compute, see https://pyradiomics.readthedocs.io/

    :return: list of metapoints with computed radiomics features
    """
    settings = {}
    # Resize mask if there is a size mismatch between image and mask
    settings["setting"] = {"correctMask": True}
    # Set the minimum number of dimensions for a ROI mask. Needed to avoid error, as in our MMG datasets we have some masses with dim=1.
    # https://pyradiomics.readthedocs.io/en/latest/radiomics.html#radiomics.imageoperations.checkMask
    settings["setting"] = {"minimumROIDimensions": 1}

    # Set feature classes to compute
    settings["featureClass"] = {feature: [] for feature in features_to_compute}
    # Suppress warnings and logging infos from radiomics
    logging.getLogger("radiomics").setLevel(logging.ERROR)

    extractor = featureextractor.RadiomicsFeatureExtractor(settings)

    logging.info("Computing radiomics for metadata")

    logging.info(
        "Enabled radiomics feature classes: {} ".format(extractor.enabledFeatures)
    )

    # Load training config file to reproduce the same preprocessing settings
    config_dict = load_yaml(path=config_path)
    config = from_dict(ClassifierConfig, config_dict)

    # Save temp metadata file. Dataset class doesn't support direct metadata input yet.
    metadata_path = Path("/tmp/tmp_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_list_of_dict = [dataclasses.asdict(metapoint) for metapoint in metadata]
    with open(metadata_path.resolve(), "w") as outfile:
        json.dump(metadata_list_of_dict, outfile, indent=4)

    # Load all metadata as a dataset without any filtering
    dataset = MammographyDataset(
        metadata_path=metadata_path,
        transform=None,
        config=config,
        filter_metadata=False,
    )

    for i, metapoint in enumerate(tqdm(metadata)):
        # Compute radiomics only for not healthy images
        if not metapoint.is_healthy:
            _, _, image, _, _, mask = dataset[i]
            sitk_image = sitk.GetImageFromArray(image)
            sitk_mask = sitk.GetImageFromArray(mask)

            output = extractor.execute(sitk_image, sitk_mask, label=255)

            radiomics_features = {}
            for feature_name in output.keys():
                # Discard non-relevant diagnostics information
                if "diagnostics" not in feature_name:
                    # Keep only shortened feature names
                    radiomics_features[feature_name.replace("original_", "")] = float(
                        output[feature_name]
                    )
        else:
            radiomics_features = None

        metapoint.radiomics = radiomics_features

    return metadata
