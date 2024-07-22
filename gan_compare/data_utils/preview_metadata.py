import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Tuple
import logging
from typing import Optional
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from dacite import from_dict
from gan_compare.training.io import load_json

from gan_compare.data_utils.utils import (
    get_image_from_metapoint,
    get_mask_from_metapoint,
)
from gan_compare.dataset.metapoint import Metapoint

# python -m gan_compare.data_utils.preview_metadata --dataset cbis-ddsm --subset train --split_path setup/train_test_val_split_ext.json --cropping --omit_healthy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_path",
        default="setup/all_metadata_w_cbis_ddsm.json",
        #required=True,
        help="Path to json file with metadata."
    ),
    parser.add_argument(
        "--dataset",
        type=str,
        default=["inbreast", "bcdr", "cbis-ddsm"],
        nargs="+",
        help="Name of the dataset of interest.",
    ),

    parser.add_argument(
        "--subset",
        type=str,
        default=None, # setup/train_test_val_split_ext.json
        help="subset can be one of 'train', 'val', 'test'.",
    ),

    parser.add_argument(
        "--split_path",
        type=str,
        default=None,
        help="The path to the file containing the train, val, test split.",
    ),

    parser.add_argument(
        "--cropping",
        action="store_true",
        help="Instead of full images we only visualize the cropped regions of interest.",
    ),

    parser.add_argument(
        "--omit_healthy",
        action="store_true",
        help="In this case, we don't want healthy patches but only cropped regions of interest.",
    ),

    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Images will be resized to (resize, resize).",
    ),
    args = parser.parse_args()
    return args


def get_crops_around_bbox(
    bbox: Tuple[int, int, int, int],
    margin: int,
    min_size: int,
    image_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox

    x_p, w_p = get_measures_for_crop(
        x, w, margin, min_size, image_shape[1]
    )
    y_p, h_p = get_measures_for_crop(
        y, h, margin, min_size, image_shape[0], w_p
    )  # second dimension depends on length of first dimension
    return (x_p, y_p, w_p, h_p)


def get_measures_for_crop(
    coord,
    length,
    margin,
    min_length,
    image_max,
        length_of_other_dimension=None,
):  # coordinate, length, margin, minimum length, random translation, random zoom
    if length_of_other_dimension is None:
        # Add a margin to the crop:
        l_new = length + 2 * margin  # add one margin left and right each

    else:
        # here we want to set the length and coordinate in relation to the other dimension, to preserve the image ratio
        # We don't add a margin but set l_new to the same value as the other dimension
        coord -= (length_of_other_dimension - length) // 2  # required to keep the patch centered
        l_new = length_of_other_dimension

    # Now make sure that the new length is at least minimum length
    if (
        l_new < min_length
    ):  # => new length is too small, must be at least minimum length
        # explanation: (min_length - l) // 2 > m
        c_new = (
            coord - (min_length - length) // 2
        )  # in this case divide by 2 to keep patch centered
        l_new = min_length
    else:  # => new length is large enough
        c_new = coord - margin

    # Now make sure that the crop is still within the image:
    c_new = max(0, c_new)
    c_new -= max(
        0, (c_new + l_new) - image_max
    )  # move crop back into the image if it goes beyond the image
    return (c_new, l_new)


if __name__ == "__main__":
    args = parse_args()
    metadata = None
    metadata_path = args.metadata_path
    assert Path(metadata_path).is_file(), f"Metadata not found in {metadata_path}"
    with open(metadata_path, "r") as metadata_file:
        metadata = [
            from_dict(Metapoint, metapoint) for metapoint in json.load(metadata_file)
        ]
    toggle = True
    if args.subset is not None and args.split_path is not None:
        split_dict = load_json(args.split_path)
        patient_ids = split_dict[args.subset]
    for metapoint in metadata:
        image_path = metapoint.image_path
        if args.omit_healthy and metapoint.is_healthy:
            continue # don't show healthy patches
        if metapoint.dataset not in args.dataset:
            continue # only specific datasets
        if args.subset is not None and args.split_path is not None and metapoint.patient_id not in patient_ids:
            continue # only ones that are in a subset (e.g. only test)
        # Getting full MMG image with corresponding mask
        image = get_image_from_metapoint(metapoint)
        mask = get_mask_from_metapoint(metapoint)

        # In case we only want to see crops, we adjust image and mask
        if args.cropping:
            x, y, w, h = get_crops_around_bbox(
                bbox=metapoint.bbox,
                min_size=224,
                image_shape=image.shape,
                margin=0 if metapoint.is_healthy else 20,
            )
            image, mask = image[y: y + h, x: x + w], mask[y: y + h, x: x + w]

        image_masked = image * (1 - mask)
        fig = plt.figure()
        fig.set_dpi(300)
        ax = plt.subplot(121)
        ax.axis("off")
        fig.suptitle(
            metapoint.dataset.upper() + " " + str(metapoint.patient_id),
            fontsize=15,
            fontweight="bold",
        )
        info_x = image.shape[1] * 1.2
        info_y = 0
        text = "Metadata: \n"
        for field in fields(metapoint):
            key = field.name
            value = getattr(metapoint, field.name)
            text += key + ": " + "%.24s" % str(value) + "\n"
        ax.text(info_x, info_y, text, fontsize=9, va="top", linespacing=2)

        bbox = metapoint.bbox
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

        if args.resize is not None:
            image = cv2.resize(image, (args.resize, args.resize), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (args.resize, args.resize), interpolation=cv2.INTER_AREA)

        if toggle:
            display = plt.imshow(image, cmap='gray')
            ax.set_title(
                "View: " + str(metapoint.laterality) + "_" + str(metapoint.view)
            )
        else:
            display = plt.imshow(image_masked)
            ax.set_title(
                "View: "
                + str(metapoint.laterality)
                + "_"
                + str(metapoint.view)
                + "_MASKED"
            )

        def onclick(event):
            global toggle
            toggle = not toggle
            if toggle:
                display.set_data(image)
                ax.set_title(
                    "View: " + str(metapoint.laterality) + "_" + str(metapoint.view)
                )
            else:
                display.set_data(image_masked)
                ax.set_title(
                    "View: "
                    + str(metapoint.laterality)
                    + "_"
                    + str(metapoint.view)
                    + "_MASKED"
                )
            event.canvas.draw()

        fig.canvas.mpl_connect("button_press_event", onclick)

        plt.show()



