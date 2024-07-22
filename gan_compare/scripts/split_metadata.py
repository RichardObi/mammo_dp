import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from gan_compare.constants import DATASET_LIST
from gan_compare.data_utils.utils import save_split_to_file
from gan_compare.dataset.constants import DENSITY_DICT


#python3 -m gan_compare.scripts.split_metadata --metadata_path setup/all_metadata_w_cbis_ddsm.json --train_proportion 0.5 --val_proportion 0.25
#python3 -m gan_compare.scripts.split_metadata --metadata_path setup/all_metadata_w_cbis_ddsm.json --train_proportion 0.4 --val_proportion 0.3 --seed 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_path",
        required=True,
        type=str,
        help="Path to json file with metadata.",
    )
    parser.add_argument(
        "--train_proportion",
        default=0.7,
        type=float,
        help="Proportion of train subset.",
    )
    parser.add_argument(
        "--val_proportion", default=0.15, type=float, help="Proportion of val subset."
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="Directory to save a split file with patient ids.",
    )
    parser.add_argument(
        "--ddsm_train_proportion",
        default=0.85,
        type=float,
        help="Proportion of train subset.",
    )
    parser.add_argument(
        "--use_provided_cbis_ddsm_split",
        action="store_true",
        help="Whether to use the train-test split that is predefined for the cbis-ddsm dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed.",
    )
    args = parser.parse_args()
    return args


def split_df_into_folds(
    metadata_df: pd.DataFrame, proportion: float
) -> Tuple[pd.DataFrame]:
    fold2_index = np.random.uniform(size=len(metadata_df)) > proportion
    fold2 = metadata_df[fold2_index]
    fold1 = metadata_df[~fold2_index]
    return fold1, fold2


def split_array_into_folds(
    patients_list: np.ndarray, proportion: float
) -> Tuple[np.ndarray]:
    fold2_index = np.random.uniform(size=len(patients_list)) > proportion
    fold2 = patients_list[fold2_index]
    fold1 = patients_list[~fold2_index]
    return fold1, fold2


if __name__ == "__main__":
    args = parse_args()
    out_path = (
        Path(args.metadata_path).parent / "train_test_val_split_ext.json"
        if args.output_path is None
        else Path(args.output_path)
    )
    np.random.seed(args.seed)
    metadata = None
    all_metadata_df = pd.read_json(args.metadata_path)
    print(f"Number of metapoints: {len(all_metadata_df)}")
    # TODO add option to append existing metadata
    train_patients_list = []
    val_patients_list = []
    test_patients_list = []
    used_patients = []
    for dataset in DATASET_LIST:  # Split is done separately for each dataset
        dataset_metadata_df = all_metadata_df[all_metadata_df.dataset == dataset]
        if len(dataset_metadata_df) == 0:
            print(f"Skipping {dataset} metadata as its empty")
            continue

        if dataset == "cbis-ddsm" and args.use_provided_cbis_ddsm_split:
            # get only test data from ddsm
            test_dataset_metadata_df = dataset_metadata_df[dataset_metadata_df.image_path.apply(lambda x: "test" in x.lower())]
            # test_patients_per_density is needed further below:
            # We do want a predefined rather than a random test split for ddsm here.
            test_patients_per_density = test_dataset_metadata_df.patient_id.unique()
            # cbis-ddsm mass metadata df needed further below.
            metadata_df_masses_ddsm = test_dataset_metadata_df[
                test_dataset_metadata_df.roi_type.apply(lambda x: "mass" in x)
            ]
            # get only training data from ddsm:
            # for training data we want a train-validation split (no test split, as test data is predefined)
            train_dataset_metadata_df = dataset_metadata_df[dataset_metadata_df.image_path.apply(lambda x: "training" in x.lower())]
            metadata_df_masses = train_dataset_metadata_df[
                dataset_metadata_df.roi_type.apply(lambda x: "mass" in x)
            ]
            metadata_df_healthy = train_dataset_metadata_df[
                dataset_metadata_df.roi_type.apply(lambda x: "healthy" in x)
            ]
            metadata_df_other_lesions = train_dataset_metadata_df[
                dataset_metadata_df.roi_type.apply(
                    lambda x: "mass" not in x and "healthy" not in x
                )
            ]
        else:
            metadata_df_masses = dataset_metadata_df[
                dataset_metadata_df.roi_type.apply(lambda x: "mass" in x)
            ]
            metadata_df_healthy = dataset_metadata_df[
                dataset_metadata_df.roi_type.apply(lambda x: "healthy" in x)
            ]
            metadata_df_other_lesions = dataset_metadata_df[
                dataset_metadata_df.roi_type.apply(
                    lambda x: "mass" not in x and "healthy" not in x
                )
            ]
        for metadata_df in [
            metadata_df_masses,
            metadata_df_healthy,
            metadata_df_other_lesions,
        ]:  # split healthy, masses and other separately to enforce balance
            for density in DENSITY_DICT.keys():
                metadata_per_density = metadata_df[metadata_df.density == density]
                patients = metadata_per_density.patient_id.unique()
                if dataset == "cbis-ddsm" and args.use_provided_cbis_ddsm_split:
                    train_patients_per_density, val_patients_per_density = split_array_into_folds(
                        patients, args.ddsm_train_proportion
                    )
                else:
                    train_patients_per_density, remaining_patients = split_array_into_folds(
                        patients, args.train_proportion
                    )
                    val_proportion_scaled = args.val_proportion / (
                        1.0 - args.train_proportion
                    )
                    (
                        val_patients_per_density,
                        test_patients_per_density,
                    ) = split_array_into_folds(remaining_patients, val_proportion_scaled)

                train_patients = train_patients_per_density.tolist()
                val_patients = val_patients_per_density.tolist()
                test_patients = test_patients_per_density.tolist()
                train_patients = [
                    train_patient
                    for train_patient in train_patients
                    if train_patient not in used_patients
                ]
                used_patients.extend(train_patients)
                train_patients_list.extend(train_patients)
                val_patients = [
                    val_patient
                    for val_patient in val_patients
                    if val_patient not in used_patients
                ]
                used_patients.extend(val_patients)
                val_patients_list.extend(val_patients)
                if dataset == "cbis-ddsm" and args.use_provided_cbis_ddsm_split:
                    # we will have multiple times the same test patients in cbis-ddsm due to current implementation (see test_patients_per_density above)
                    # therefore, we do not add test_patient if already in final test_patients_list to avoid duplicates
                    test_patients = [
                        test_patient
                        for test_patient in test_patients
                        if test_patient not in used_patients and test_patient not in test_patients_list
                    ]
                else:
                    test_patients = [
                        test_patient
                        for test_patient in test_patients
                        if test_patient not in used_patients
                    ]
                used_patients.extend(test_patients)
                test_patients_list.extend(test_patients)
        masses_train = metadata_df_masses[
            metadata_df_masses.patient_id.apply(lambda x: x in train_patients_list)
        ]
        print(f"Masses in {dataset} train: {len(masses_train.index)}")

        masses_val = metadata_df_masses[
            metadata_df_masses.patient_id.apply(lambda x: x in val_patients_list)
        ]
        print(f"Masses in {dataset} val: {len(masses_val.index)}")

        if dataset == "cbis-ddsm" and args.use_provided_cbis_ddsm_split:
            masses_test = metadata_df_masses_ddsm[
                metadata_df_masses_ddsm.patient_id.apply(lambda x: x in test_patients_list)
            ]
        else:
            masses_test = metadata_df_masses[
                metadata_df_masses.patient_id.apply(lambda x: x in test_patients_list)
            ]
        print(f"Masses in {dataset} test: {len(masses_test.index)}")

    # Some metapoints may not contain density label - we don't want them in any of the splits
    assert (
        len(train_patients_list) + len(val_patients_list) + len(test_patients_list)
    ) == len(used_patients)
    # TODO calculate and print statistics of subsets in terms of lesion types, dataset types, densities
    print(
        f"Split patients into {len(train_patients_list)}, {len(val_patients_list)} and {len(test_patients_list)} samples."
    )

    # Double-checking that no patient_id is in more than one of training, validation, and test.
    for patient_id in train_patients_list:
        assert(patient_id not in val_patients_list), f"At least one patient_id ({patient_id}) was found to be in both training and validation sets. Revise with caution."
        assert(patient_id not in test_patients_list), f"At least one patient_id ({patient_id}) was found to be in both training and test sets. Revise with caution."
    for patient_id in train_patients_list:
        assert(patient_id not in test_patients_list), f"At least one patient_id ({patient_id}) was found to be in both validation and test sets. Revise with caution."

    print("Saving..")
    save_split_to_file(
        train_patient_ids=train_patients_list,
        val_patient_ids=val_patients_list,
        test_patient_ids=test_patients_list,
        out_path=out_path,
    )

    print(f"Saved a split file with patient ids to {out_path.resolve()}")
