import logging
import os
from dataclasses import dataclass, field
from time import strftime
from typing import Dict, List, Optional


from gan_compare.dataset.constants import RADIOMICS_TARGET_FEATURE_LIST
from gan_compare.training.dataset_config import DatasetConfig
from gan_compare import constants


@dataclass
class BaseConfig:

    metadata_path: str
    data: Dict[str, DatasetConfig]

    logfile_path: Optional[str] = None # Changing this only works for train_test_split.py at the moment and not for gan training.

    log_level: int = logging.INFO # The log level as a measure of severity

    # Paths to train and validation metadata
    split_path: str = None

    # model type and task (CLF / GAN) not known. Overwritten in specific configs
    model_name: str = "unknown"

    # Default training target

    # The training objective from where classes are derived.
    # training target and classes are in each metapoint in the dataset
    training_target: str = "biopsy_proven_status"

    # The classes possible in the training_target field in the metapoints.
    # These classes are either used for classification or for GAN conditioning.
    classes: str = "is_benign"  # one of ["is_benign", "is_healthy", "birads"]

    # Specify whether the training task is regression or not (i.e. classification)
    is_regression: bool = False

    # The selected radiomics features that should be used for classification pretraining
    target_radiomics_features: List[str] = field(
        default_factory=lambda: RADIOMICS_TARGET_FEATURE_LIST
    )

    train_sampling_ratio: float = 1.0

    # Overwritten in specific configs
    output_model_dir: str = ""

    # Birads range
    birads_min: int = 2
    birads_max: int = 6

    # random seed
    seed: int = 42
    # 4a 4b 4c of birads are splitted into integers
    split_birads_fours: bool = True
    # The number of condition labels for input into conditional GAN (i.e. 7 for BI-RADS 0 - 6)
    # OR for classification, the number of classes (set automatically though in classification_config.py)
    n_cond: int = birads_max + 1
    # Number of workers for dataloader
    workers: int = 2
    # Batch size during training
    batch_size: int = 8
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size: int = 64
    # Number of training epochs
    num_epochs: int = 60
    # Learning rate for optimizers
    lr: float = 0.0001  # Used only in train_test_classifier.py. Note: there are lr_g, lr_d in gan_config.py
    # number of GPUs
    ngpu: int = 1
    # Whether to train conditional GAN
    conditional: bool = False
    # We can condition on different variables such as breast density or birads status of lesion. Default = "density"
    conditioned_on: str = None  # "density", "birads"
    # Specify whether condition is modeled as binary e.g., benign/malignant with birads 1-3 = 0, 4-6 = 1
    is_condition_binary: bool = False

    # Preprocessing of training images
    # Variables for utils.py -> get_measures_for_crop():
    zoom_offset: float = 0.2  # the higher, the more likely the patch is zoomed out. if 0, no offset. negative means, patch is rather zoomed in
    zoom_spread: float = 0.33  # the higher, the more variance in zooming. Must be greater or equal 0. with 0. being minimal variance.
    ratio_spread: float = 0.05  # NOT IN USE ANYMORE. coefficient for how much to spread the ratio between height and width. the higher, the more spread.
    translation_spread: float = 0.25  # the higher, the more variance in translation. Must be greater or equal 0. with 0. being minimal variance.
    max_translation_offset: float = 0.33  # coefficient relative to the image size.

    #### Variables for differential privacy setup both usable for GAN and classifier training.
    use_dp: bool = False # switch that decides whether DP should be enabled or not during training

    # Epsilon measures the privacy loss in Differential Privacy (smaller epsilon = more privacy), i.e. try 0.1 to 50.",
    # You can play around with the level of privacy, EPSILON. Smaller EPSILON means more privacy, more noise -- and hence lower accuracy.
    # Reducing EPSILON to 5.0 reduces the Top 1 Accuracy to around 53%. One useful technique is to pre-train a model on public (non-private) data, before completing the training on the private training data. See the workbook at bit.ly/opacus-dev-day for an example.
    # Example: https://opacus.ai/tutorials/building_image_classifier
    dp_target_epsilon: float = 50.0

    # Delta represents the probability that a range of outputs with a privacy loss >epsilon exists in Differential Privacy. (smaller delta = more privacy), i.e. 10^−5 is a common value",
    # The target δ of the (ϵ,δ)-differential privacy guarantee. Generally, it should be set to be less than the inverse of the size of the training dataset.
    # It is set to $10^{−5}$ as the CIFAR10 dataset has 50,000 training points.
    # Our cbis-ddsm has roughly 1k training points, so we set delta to 0.0001 i.e. 1e−4
    dp_target_delta: float = 0.0001 # 0.00001

    # The maximum L2 norm of per-sample gradients before they are aggregated by the averaging step. ",
    # The maximum norm of the per-sample gradients. Any gradient with norm higher than this will be clipped to this value.
    # Tuning MAX_GRAD_NORM is very important e.g. do a grid search for the optimal MAX_GRAD_NORM value. The grid can be in the range [.1, 10].
    dp_max_grad_norm: float = 1. # 10.


    def __post_init__(self):
        if self.model_name in constants.swin_transformer_names:
            self.image_size: int = (
                224  # swin transformer currently only supports 224x224 images
            )
            self.nc: int = 3
            logging.info(
                f"Changed image shape to {self.image_size}x{self.image_size}x{self.nc}, as is needed for the selected model ({self.model_name})"
            )
        if self.output_model_dir is None or self.output_model_dir == '':
            self.output_model_dir = os.path.join(
                self.output_model_dir,
                f"training_{self.model_name}_{strftime('%Y_%m_%d-%H_%M_%S')}",
            )
