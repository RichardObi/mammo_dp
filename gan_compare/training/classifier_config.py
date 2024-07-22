import logging
from dataclasses import dataclass

import ijson
import torch.nn as nn
from dacite import from_dict
from typing import Optional

from gan_compare.dataset.metapoint import Metapoint
from gan_compare.training.base_config import BaseConfig


@dataclass
class ClassifierConfig(BaseConfig):

    train_shuffle_proportion: float = 0.4
    validation_shuffle_proportion: float = 0

    # Directory with synthetic patches
    synthetic_data_dir: Optional[str] = None

    # Proportion of training artificial images
    gan_images_ratio: float = 0.5

    no_transforms: bool = False

    # Whether to use synthetic data at all
    use_synthetic: bool = True

    # Dropout rate
    dropout_rate: float = 0.3

    out_checkpoint_path: str = ""

    optimizer_type: str ="sgd" # "adamw"

    # label smoothing
    clf_label_smoothing: float = 0.0 #0.1#

    # weight decay in adam and adamw
    clf_weight_decay: float = 0.0 # 0.00000001

    # learning rate
    clf_lr:float = 0.001

    # Metapoint attribute to be used as a training target
    # Note: some attributes have special scenarios based on additional configuration
    training_target: Optional[str] = None

    # Default loss in case of regression training
    regression_loss = nn.L1Loss()

    # indicate which of the radimagenet swin-t weights to choose
    radimagenet_weights: int = 1

    # Learning rate for optimizer
    lr: float = 0.0001  # Note: The CLF equivalent of the learning rates lr_g, lr_d1, lr_d2 in gan_config.py for GAN training.

    # Which format to use when outputting the classification results on the test set, either json, csv, or None. If None, no such results are output.
    output_classification_result: Optional[str] = None

    # Which type of pre-training we want to use here. If None, no pre-trained weights are used. Possible modes: "freeze_almost_all_weights" and "freeze_no_weights".
    pretrained_mode: Optional[str] = None

    # Name of the template specifying which layers to freeze (c.f. gan_compare/constants.py for available templates, e.g. "freeze_template_swin_transformer_almost_all_layers").
    pretrained_freeze_template: Optional[str] = None

    # Path to the pretrained weights; used if pretrained_mode is set accordingly.
    pretrained_weights_path: Optional[str] = None

    # Do we have binary classification or rather a multiclass CLF
    binary_classification: Optional[bool] = None

    # Number of classification target classes
    num_classes: int = 2

    reinit_head:bool = None

    def __post_init__(self):
        super().__post_init__()
        #if not len(self.out_checkpoint_path) > 0:
        self.out_checkpoint_path = f"{self.output_model_dir}/best_classifier.pt"

        assert self.classes in [
            "is_benign",
            "is_healthy",
            "birads",  # FIXME metrics.py needs to be extended to handle multiclass CLF.
        ]

        (
            self.is_regression,
            self.num_classes
        ) = self.deduce_training_target_task_and_size()

        if (self.num_classes == 1 or self.classes in ["is_benign", "is_healthy"]) and not self.is_regression:
            self.binary_classification = True
        else:
            self.binary_classification = False

        if self.binary_classification:
            self.n_cond = 2
        elif self.split_birads_fours:
            self.birads_min = 1
            self.birads_max = 7
            self.n_cond = self.birads_max + 1

        assert (
            1
            >= self.train_shuffle_proportion
            >= 0  # it looks like it is not used anywhere, it is quite misleading
        ), "Train shuffle proportion must be from <0,1> range"
        assert (
            1 >= self.validation_shuffle_proportion >= 0
        ), "Validation shuffle proportion must be from <0,1> range"

        self.loss = self.deduce_loss()

    def deduce_training_target_task_and_size(self):

        num_classes:int = 1
        is_regression = False


        if self.training_target in ["is_benign", "is_healthy", "biopsy_proven_status", "healthy"]:
            # Binary classification, default values are used
            num_classes:int = 2
            pass
        else:

            try:
                metadata_file = open(self.metadata_path, "r")
                metadata = ijson.items(metadata_file, "item")
                target = None
                for metapoint in metadata:
                    metapoint = from_dict(Metapoint, metapoint)
                    target = getattr(metapoint, self.training_target)
                    if target is not None:
                        break

                if type(target) == int:
                    num_classes:int = 1
                    is_regression = True
                elif type(target) == str:
                    if self.training_target == "birads":
                        num_classes:int = self.n_cond
                    else:
                        # Getting two logits for CrossEntropy Loss as in https://discuss.pytorch.org/t/crossentropy-in-pytorch-getting-target-1-out-of-bounds/71575/2
                        num_classes:int = 2
                elif type(target) == dict:
                    is_regression = True
                    if self.training_target == "radiomics":
                        num_classes:int = self.validate_and_set_target_radiomics_features(
                            target=target
                        )
                    else:
                        num_classes:int = len(target.items())
                elif type(target) == bool:
                    num_classes:int = 2

                elif target is None:
                    raise ValueError(
                        "No values found for target {}".format(self.training_target)
                    )

            except AttributeError:
                raise AttributeError(
                    "Target {} not found in config nor in metadata".format(
                        self.training_target
                    )
                )
        return is_regression, num_classes

    def validate_and_set_target_radiomics_features(self, target) -> int:
        if self.target_radiomics_features is None:
            logging.warning(
                f"You did not specify 'target_radiomics_features' ({self.target_radiomics_features}): All {len(target.items())} {self.training_target} features are selected."
            )
            return len(target.items())
        else:
            assert (
                len(self.target_radiomics_features) > 0
            ), f"Please provide at least 1 {self.training_target} feature in config variable 'target_radiomics_features' ({self.target_radiomics_features})"

            # Merging available radiomics targets and use-specified targets
            training_classes = list(target.keys()) + list(
                self.target_radiomics_features
            )

            # (1) Only keep the duplicates that are both in metadata and in our target_radiomics_features (2) and then remove duplicates
            training_classes = list(
                set([i for i in training_classes if training_classes.count(i) > 1])
            )

            missed_features = list(
                set(self.target_radiomics_features).difference(training_classes)
            )
            if len(missed_features) > 0:
                logging.warning(
                    f"You provided radiomics features that were not in the metadata and therefore will be discarded: {missed_features}"
                )
            self.target_radiomics_features = training_classes
            logging.info(
                f"Setup of {self.training_target} training: {len(training_classes)} features will be predicted, namely: {training_classes}."
            )
            return len(training_classes)

    def deduce_loss(self):
        if self.is_regression:
            return self.regression_loss
        else:
            return nn.CrossEntropyLoss()
