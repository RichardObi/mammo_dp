import argparse
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import copy

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import torchvision.transforms as transforms
from dacite import from_dict
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm

from gan_compare.constants import get_classifier, make_private_with_epsilon, only_use_opacus_supported_layers
from gan_compare.data_utils.utils import init_seed, setup_logger
from gan_compare.dataset.mammo_dataset import MammographyDataset
from gan_compare.dataset.synthetic_dataset import SyntheticDataset
from gan_compare.dataset.directory_dataset import DirectoryDataset
#from gan_compare.scripts.embedder import get_top_k_from_image, get_index_from_dataloader, get_nn_dataloader, load_embedding_model

from gan_compare.scripts.metrics import (
    calc_all_scores,
    calc_AUPRC,
    calc_AUROC,
    calc_loss,
    from_logits_to_probs,
    output_ROC_curve,
)
from gan_compare.training.classifier_config import ClassifierConfig
from gan_compare.training.io import load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="gan_compare/configs/classification_config.yaml",
        help="Path to a yaml model config file",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="If this is set, the paths/metadata in the config.yaml will be ignored. "
             "Instead the images in the herein provided dataset_path will be used to create dataloaders."
             " Path to a dir containing train/val/test images.",
    )
    parser.add_argument(
        "--only_get_metrics",
        action="store_true",
        help="Whether to skip training and just evaluate the model saved in the default location.",
    )
    parser.add_argument(
        "--in_checkpoint_path",
        type=str,
        nargs='+',
        default=["model_checkpoints/classifier/classifier.pt"],
        help="Only required if --only_get_metrics is set. Path to checkpoint to be loaded.",
    )
    parser.add_argument(
        "--out_checkpoint_path",
        type=str,
        default=None,
        help="Overwrites config.out_checkpoint_path if not None.",
    )
    parser.add_argument(
        "--save_dataset",
        action="store_true",
        help="Whether to save image patches to images_classifier dir",
    )
    parser.add_argument(
        "--save_per_epoch",
        action="store_true",
        help="Whether to store the model after each epoch.",
    )

    parser.add_argument(
        "--save_each_n_epochs",
        default=299,
        type=int,
        help="If the model is not stored each epoch, then store it each n_th epoch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        help="Device to be used for training/testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--train_sampling_proportion",
        type=float,
        default=None,
        help="The proportion of the images in the training dataset that are actually used during training.",
    )
    parser.add_argument(
        "--use_dp",
        action="store_true",
        help="The switch that decides whether DP should be enabled or not during training.",
    )
    parser.add_argument(
        "--density_filter",
        action="store_true",
        help="The switch that starts breast density specific lesion filtering during training (only 1+2) and separate tests per density class group (e.g. 1+2 vs 3+4).",
    )
    parser.add_argument(
        "--train_densities",
        type=list,
        default=[1, 2],
        help="The density types used for training and splitted in testing.",
    )
    parser.add_argument(
        "--all_densities",
        type=list,
        default=[1, 2, 3, 4],
        help="All possible density types used for training, validation and testing.",
    )
    parser.add_argument(
        "--mass_margin_filter",
        action="store_true",
        help="The switch that starts mass margin specific lesion filtering during training (e.g. only 1+2) and separate tests per class group (e.g. 1+2 vs 3+4).",
    )
    parser.add_argument(
        "--train_mass_margins",  # everything but these are used for training
        type=list,
        default=["obscured", "ill_defined"],
        help="The mass margin types used for training and splitted in testing.",
    )
    parser.add_argument(
        "--mass_subtlety_filter",
        action="store_true",
        help="The switch that starts mass susceptibility specific lesion filtering during training (e.g. only 1+2) and separate tests per class group (e.g. 1+2 vs 3+4).",
    )
    parser.add_argument(
        "--train_subtleties",  # everything but these are used for training
        type=list,
        default=[5],  # [0,1,2,3],
        help="The mass susceptibility (e.g. 1 to 5) used for training and splitted in testing.",
    )
    parser.add_argument(
        "--all_subtleties",
        type=list,
        default=[0, 1, 2, 3, 4, 5],
        help="All possible subtlety types used for training, validation and testing.",
    )
    parser.add_argument(
        "--ensemble_strategies",
        type=list,
        default=["1"],
        help="1=Voting as average of model predictions, 2=Voting as weighted average of model predictions weighted by metric on validation set,  3=Majority voting where all models have equal voting weight, 4=Majority voting where each model's vote is weighted by validation loss, 5=Voting as weighted average of model predictions weighted by metric over the top k validation set nearest neighbors of each test sample.",
    )
    parser.add_argument(
        "--ensemble_weight_multiplier",
        type=str,
        default="val_loss",
        help="The validation set metric for weighting models in ensemble. One of: [val_loss, val_loss^2, val_acc, val_auroc, val_auprc, val_auprc",
    )

    parser.add_argument(
        "--weighting_by_rank",
        action="store_true",
        help="If the ensemble is weighted, we can either weight by model rank (e.g. best val_acc => rank 1 => weight multipler = 10*num_models), or, without ranking directly multiply by validation metric (e.g. val_acc = weight multipler)",
    )

    parser.add_argument(
        "--validation_set",
        action="store_true",
        help="If this flag is set to true and only_get_metrics is also true, we evaluate on the validation set instead of on the test set.",
    )

    parser.add_argument(
        "--embedding_model_type",
        type=str,
        default="inceptionv3",
        help="In case embeddings are generated (e.g. to weight ensembles via ANN), then this model is used to create embeddings. Choose from: 'config' 'swin' 'inceptionv3' 'vgg16' 'resnet50'.",
    )

    parser.add_argument(
        "--top_k",
        type=str,
        default="10",
        help="Number of returned indices in approx nn search.",
    )
    parser.add_argument(
        "--remove_all_but_l_models",
        type=str,
        default=None,
        help="Number l (e.g. set to '10') of remaining models. models are removed with strategy 5.",
    )
    parser.add_argument(
        "--use_training_in_ann",
        action="store_true",
        help="Use the training dataset instead of the validation dataset to select/weight best models",
    )
    parser.add_argument(
        "--num_models_to_remove",
        type=int,
        default=None,
        help="Number of models in the ensemble to be removed in each batch when iterating over test set batches.",
    )
    parser.add_argument(
        "--min_num_models",
        type=int,
        default=10,
        help="If models are to be removed, then the number of models in the ensemble cannot be less than min_num_models.",
    )

    parser.set_defaults(use_dp=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Parse config file
    try:
        config_dict = load_yaml(path=args.config_path)
    except:
        full_config_path = os.path.join(os.getcwd(),
                                        args.config_path)  # in case relative paths are given and don't work
        config_dict = load_yaml(path=full_config_path)
    config = from_dict(ClassifierConfig, config_dict)
    config.logfile_path = config.logfile_path.replace(".", f"_seed{args.seed}.")
    logfilename, logfile_path = setup_logger(logfile_path=config.logfile_path, log_level=config.log_level)

    # FIXME For fast testing:
    #config.num_epochs = 1

    if args.out_checkpoint_path is not None:
        config.out_checkpoint_path = args.out_checkpoint_path
    if not config.out_checkpoint_path.endswith(".pt"):
        if args.save_each_n_epochs is None and not args.save_per_epoch:
            config.out_checkpoint_path += (
                f'{datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}/classifier.pt'
            )
        else:
            config.out_checkpoint_path = os.path.join(config.out_checkpoint_path, f'classifier.pt')

    logging.info(str(asdict(config)))
    # logging.info("Classifier LR: "+ str(config.clf_lr))
    logging.info(str(args))
    logging.info(
        "Loading dataset..."
    )  # When we have more datasets implemented, we can specify which one(s) to load in config.

    init_seed(args.seed)  # setting the seed from the args

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and config.ngpu > 0 and args.device != "cpu") else "cpu"
    )
    logging.info(f"Device: {device}")

    criterion = get_criterion(config)

    logging.info(f"Loss function: {criterion}")

    net = get_classifier(config, device=device).to(device)
    net.to(device)

    if config.use_synthetic:
        assert (
                config.synthetic_data_dir is not None
        ), "If you want to use synthetic data, you must provide a diretory with the patches in config.synthetic_data_dir."
    if config.no_transforms:
        train_transform = transforms.Compose(
            [
                transforms.Normalize((0.5), (0.5)),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )
    val_transform = transforms.Compose(
        [
            transforms.Normalize((0.5), (0.5)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Normalize((0.5), (0.5)),
        ]
    )
    if args.train_sampling_proportion is not None:
        # overriding config value if function argument train_sampling_proportion is received.
        logging.warning(
            f"config.train_sampling_ratio ({config.train_sampling_ratio}) is overwritten with args.train_sampling_proportion ({args.train_sampling_proportion}).")
        config.train_sampling_ratio = args.train_sampling_proportion

    density_filter_train = None
    density_filter_test1 = None
    density_filter_test2 = None
    mass_margin_filter_train = None
    mass_margin_filter_test1 = None
    mass_margin_filter_test2 = None
    mass_subtlety_filter_train = None
    mass_subtlety_filter_test1 = None
    mass_subtlety_filter_test2 = None
    if args.density_filter:
        args.train_densities = [int(density) for density in args.train_densities if str(density).isnumeric()]
        density_filter_train = args.train_densities
        density_filter_test1 = args.train_densities
        density_filter_test2 = [int(density) for density in args.all_densities if
                                int(density) not in args.train_densities]
    if args.mass_margin_filter:
        mass_margin_filter_train = copy.deepcopy(args.train_mass_margins)
        # mass_margin_filter_train.append("!")
        mass_margin_filter_test1 = copy.deepcopy(args.train_mass_margins)
        # mass_margin_filter_test1.append("!")
        mass_margin_filter_test2 = copy.deepcopy(args.train_mass_margins)
        mass_margin_filter_test2.append("!")
    if args.mass_subtlety_filter:
        args.train_subtleties = [int(subtlety) for subtlety in args.train_subtleties if str(subtlety).isnumeric()]
        mass_subtlety_filter_train = args.train_subtleties
        mass_subtlety_filter_test1 = args.train_subtleties
        mass_subtlety_filter_test2 = [int(subtlety) for subtlety in args.all_subtleties if
                                      int(subtlety) not in args.train_subtleties]

    #If we want to load the dataset from a directory instead of the config file, we do it here.
    if args.dataset_path is not None:
        logging.info(f"Loading dataset from directory rather than metadata. Directory root: {args.dataset_path}")
        train_dataset = DirectoryDataset(
            config=config,  # e.g. used to determine which dataset to use (e.g. cbis-ddsm, bcdr), and image size
            subset="train",
            transform = train_transform,
            dataset_path = args.dataset_path,
        )
        val_dataset = DirectoryDataset(
            config=config,  # e.g. used to determine which dataset to use (e.g. cbis-ddsm, bcdr), and image size
            subset="val",
            transform = val_transform,
            dataset_path = args.dataset_path,
        )
        test_dataset1 = DirectoryDataset(
            config=config,  # e.g. used to determine which dataset to use (e.g. cbis-ddsm, bcdr), and image size
            subset="test",
            transform = test_transform,
            dataset_path = args.dataset_path,
            dataset_name="cbis-ddsm" # overrides the dataset specified in the config
        )
        test_dataset2 = DirectoryDataset(
            config=config,  # e.g. used to determine which dataset to use (e.g. cbis-ddsm, bcdr), and image size
            subset="test",
            transform = test_transform,
            dataset_path = args.dataset_path,
            dataset_name="bcdr" # overrides the dataset specified in the config
        )
    else:
        train_dataset = MammographyDataset(
            metadata_path=config.metadata_path,
            split_path=config.split_path,
            subset="train",
            config=config,
            sampling_ratio=config.train_sampling_ratio,
            transform=train_transform,
            density_filter=density_filter_train,
            mass_margin_filter=mass_margin_filter_train,
            mass_subtlety_filter=mass_subtlety_filter_train,
        )
        val_dataset = MammographyDataset(
            metadata_path=config.metadata_path,
            split_path=config.split_path,
            subset="val",
            config=config,
            transform=val_transform,
            density_filter=density_filter_train,
            mass_margin_filter=mass_margin_filter_train,
            mass_subtlety_filter=mass_subtlety_filter_train,

        )
        if args.density_filter or args.mass_margin_filter or args.mass_subtlety_filter:
            test_dataset1 = MammographyDataset(
                metadata_path=config.metadata_path,
                split_path=config.split_path,
                subset="test",
                config=config,
                transform=test_transform,
                density_filter=density_filter_test1,
                mass_margin_filter=mass_margin_filter_test1,
                mass_subtlety_filter=mass_subtlety_filter_test1,
                return_indices=str(5) in args.ensemble_strategies,
            )
            test_dataset2 = MammographyDataset(
                metadata_path=config.metadata_path,
                split_path=config.split_path,
                subset="test",
                config=config,
                transform=test_transform,
                density_filter=density_filter_test2,
                mass_margin_filter=mass_margin_filter_test2,
                mass_subtlety_filter=mass_subtlety_filter_test2,
                return_indices=str(5) in args.ensemble_strategies,
            )
        else:
            test_dataset = MammographyDataset(
                metadata_path=config.metadata_path,
                split_path=config.split_path,
                subset="test",
                config=config,
                transform=test_transform,
                return_indices=str(5) in args.ensemble_strategies,
            )
    train_dataset_no_synth = train_dataset
    if config.use_synthetic:
        # APPEND SYNTHETIC DATA
        synth_train_images = SyntheticDataset(
            transform=train_transform,
            config=config,
        )
        train_dataset = ConcatDataset([train_dataset, synth_train_images])
        logging.info(
            f"Number of synthetic patches added to training set: {len(synth_train_images)}"
        )

    if config.binary_classification and False:  # and False:
        num_train_negative, num_train_positive = get_label_balance_info(train_dataset, "train",
                                                                        config)  # needed for WeightedRandomSampler
        # num_val_negative, num_val_positive = get_label_balance_info(val_dataset, "val", config) # just for printing info
        # num_test_negative, num_test_positive = get_label_balance_info(test_dataset, "test", config) # just for printing info

        # Compute the weights for the WeightedRandomSampler for the training set:
        # Example: labels of training set: [true, true, false] => weight_true = 3/2; weight_false = 3/1
        weight_negative = len(train_dataset) / num_train_negative
        weight_positive = len(train_dataset) / num_train_positive
        train_weights = []

        if config.classes == "is_healthy":
            train_weights.extend(
                train_dataset_no_synth.arrange_weights_healthy(weight_negative, weight_positive)
            )
        else:
            train_weights.extend(
                arrange_weights(weight_class_positive=weight_positive, weight_class_negative=weight_negative,
                                dataset=train_dataset)
            )

        train_sampler = WeightedRandomSampler(train_weights, len(train_dataset))
    else:
        train_sampler = None

    # We don't want any sample weights in validation and test sets, so we stick with shuffle=True below.
    if len(train_dataset) > 0:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.workers,
            sampler=train_sampler,
            shuffle=True,
        )

    if len(val_dataset) > 0:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=config.workers,
            shuffle=False,
        )
    test_dataloaders = []
    if test_dataset1 is not None and test_dataset2 is not None and args.dataset_path is not None:
        test_dataloaders.append(
            DataLoader(
                test_dataset1,
                batch_size=config.batch_size,
                num_workers=config.workers,
                shuffle=False,
            )
        )
        test_dataloaders.append(
            DataLoader(
                test_dataset2,
                batch_size=config.batch_size,
                num_workers=config.workers,
                shuffle=False,
            )
        )
    elif args.density_filter or args.mass_margin_filter or args.mass_subtlety_filter:
        test_dataloaders.append(
            DataLoader(
                test_dataset1,
                batch_size=config.batch_size if not str(5) in args.ensemble_strategies else 1,
                num_workers=config.workers,
                shuffle=False,
            )
        )
        test_dataloaders.append(
            DataLoader(
                test_dataset2,
                batch_size=config.batch_size if not str(5) in args.ensemble_strategies else 1,
                num_workers=config.workers,
                shuffle=False,
            )
        )
    else:
        test_dataloaders.append(
            DataLoader(
                test_dataset,
                batch_size=config.batch_size if not str(5) in args.ensemble_strategies else 1,
                num_workers=config.workers,
                shuffle=False,
            )
        )

    if not Path(config.out_checkpoint_path).parent.exists():
        os.makedirs(Path(config.out_checkpoint_path).parent.resolve(), exist_ok=True)

    if args.save_dataset:
        # This code section is only for saving patches as image files and further info about the patch if needed.
        # The program stops execution after this section and performs no training.

        # TODO: state_dict name should probably be in config yaml instead of hardcoded.
        net.load_state_dict(
            torch.load("model_checkpoints/classifier 50 no synth/classifier.pt", map_location=device)
        )
        net.eval()
        # TODO refactor (make optional & parametrize) or remove saving dataset
        save_data_path = Path("save_dataset")

        with open(save_data_path / "validation.txt", "w") as f:
            f.write("index, y_prob\n")
        cnt = 0
        metapoints = []
        for data in tqdm(
                test_dataset
        ):  # this has built-in shuffling; if not shuffled, only lesion-containing patches will be output first
            sample, label, image, r, d = data
            outputs = net(sample[np.newaxis, ...])

            # for y_prob_logit, label, image, r, d in zip(outputs.data, labels, images, rs, ds):
            y_prob_logit = outputs.data
            y_prob = from_logits_to_probs(y_prob_logit)
            metapoint = {
                "label": label,
                "roi_type": r,
                "dataset": d,
                "cnt": cnt,
                "y_prob": y_prob.numpy().tolist(),
            }
            metapoints.append(metapoint)

            # with open(save_data_path / "validation.txt", "a") as f:
            #     f.write(f'{cnt}, {y_prob}\n')

            label = "healthy" if int(label) == 1 else "with_lesions"
            out_image_dir = save_data_path / "validation" / str(label)
            out_image_dir.mkdir(parents=True, exist_ok=True)
            out_image_path = out_image_dir / f"{cnt}.png"
            cv2.imwrite(str(out_image_path), np.array(image))

            cnt += 1
        with open(save_data_path / "validation.json", "w") as f:
            f.write(json.dumps(metapoints))

        logging.info(f"Saved data samples to {save_data_path.resolve()}")
        exit()

    if args.use_dp or config.use_dp:  # for ease of use, we want to allow activating DP both via args as well as via config. Both have default=False
        net = only_use_opacus_supported_layers(net=net)

    if not args.only_get_metrics:

        # Inputting the seed information into the checkpoint saving path.
        config.out_checkpoint_path = config.out_checkpoint_path.replace(".",
                                                                        f"_seed{str(args.seed).strip()}_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.")

        # PREPARE TRAINING
        # TODO: Optimizer params (lr, momentum) should be moved to classifier_config.
        if config.optimizer_type == "sgd":
            optimizer = optim.SGD(net.parameters(), lr=config.clf_lr, momentum=0.9)  # lr=0.001, momentum=0.9)
        elif config.optimizer_type == "adam":
            optimizer = optim.Adam(net.parameters(), lr=config.clf_lr, weight_decay=config.clf_weight_decay)
        elif config.optimizer_type == "adamw":
            optimizer = optim.AdamW(net.parameters(), lr=config.clf_lr, weight_decay=config.clf_weight_decay)

        if args.use_dp or config.use_dp:  # for ease of use, we want to allow activating DP both via args as well as via config. Both have default=False
            net, optimizer, train_dataloader, privacy_engine = make_private_with_epsilon(net=net, optimizer=optimizer,
                                                                                         dataloader=train_dataloader,
                                                                                         dp_target_epsilon=config.dp_target_epsilon,
                                                                                         dp_target_delta=config.dp_target_delta,
                                                                                         dp_max_grad_norm=config.dp_max_grad_norm,
                                                                                         num_epochs=config.num_epochs,
                                                                                         auto_validate_model=False,
                                                                                         grad_sample_mode=None,  # "ew"
                                                                                         )
            final_epsilon = float("inf")
        best_loss = float("inf")
        best_f1 = 0
        best_epoch = 0
        best_prc_auc = 0

        # START TRAINING LOOP
        for epoch in tqdm(
                range(config.num_epochs)
        ):  # loop over the dataset multiple times
            running_loss = 0.0
            y_true_train = []
            y_prob_logit_train = []
            train_loss = []
            logging.info(f"Training in epoch {epoch}... {args.config_path}")
            for i, data in enumerate(tqdm(train_dataloader)):

                # get the inputs; data is a list of [inputs, labels]
                try:
                    samples, labels, _, _, _, _ = data
                except:
                    samples, labels, _, _, _ = data

                if len(samples) <= 1:
                    continue  # batch normalization won't work if samples too small (https://stackoverflow.com/a/48344268/3692004)

                # Explicitely setting model into training mode.
                net.train()

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(samples.to(device))
                logging.debug(f"outputs (shape:{outputs.shape}): {outputs}")
                logging.debug(f"labels: {labels}")
                # Converting to float as model (e.g. swin) can return doubles leading to dtpye mismatch error.
                y_true_train.append(labels)

                # logging.info(f"outputs type: {outputs.dtype}, outputs shape: {outputs.shape}, labels type: {labels.dtype}, labels shape: {labels.shape}")

                # if outputs.shape != labels.shape:
                # labels = torch.unsqueeze(labels, dim=-1).to(device).to(torch.float32)

                try:
                    loss = criterion(outputs, labels.to(device))
                except:
                    labels = labels.type(torch.float)
                    loss = criterion(outputs, (labels.to(device)))

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    if args.use_dp or config.use_dp:  # for ease of use, we want to allow activating DP both via args as well as via config. Both have default=False
                        epsilon = privacy_engine.get_epsilon(config.dp_target_delta)
                        logging.info(
                            "[%d, %5d] loss: %.3f  DP(ε=%.3f,δ=%.3f) " % (
                            epoch + 1, i + 1, running_loss / 2000, epsilon, config.dp_target_delta)
                        )
                    else:
                        logging.info(
                            "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000)
                        )
                    running_loss = 0.0

                y_prob_logit_train.append(outputs.data.cpu())
                # train_loss.append(loss.detach().cpu())

            # Calculating metrics for training
            y_true_train = torch.cat(y_true_train)
            y_prob_logit_train = torch.cat(y_prob_logit_train)
            y_prob_train = from_logits_to_probs(y_prob_logit_train)
            _, _, prec_rec_f1, roc_auc, prc_auc = calc_all_scores(
                y_true_train,
                y_prob_train,
                train_loss,
                "Training",
                config,
                epoch,
                config_path=args.config_path,
                test_only=args.only_get_metrics,
            )

            # saving model
            if args.save_per_epoch:
                torch.save(net.state_dict(), config.out_checkpoint_path.replace("best_classifier", f"{epoch}"))
            elif epoch % args.save_each_n_epochs == 0 and epoch != 0:
                torch.save(net.state_dict(), config.out_checkpoint_path.replace("best_classifier", f"{epoch}"))

            # VALIDATE
            val_loss = []
            with torch.no_grad():
                y_true = []
                y_prob_logit = []
                net.eval()
                if args.use_dp or config.use_dp:  # for ease of use, we want to allow activating DP both via args as well as via config. Both have default=False
                    epsilon = privacy_engine.get_epsilon(config.dp_target_delta)
                    logging.info(
                        f"Validating in epoch {epoch}... {args.config_path}. Current DP -> DP(ε={epsilon}, δ={config.dp_target_delta}).")
                else:
                    logging.info(f"Validating in epoch {epoch}... {args.config_path} ")
                for i, data in enumerate(tqdm(val_dataloader)):
                    try:
                        samples, labels, _, _, _, _ = data
                    except:
                        samples, labels, _, _, _ = data
                    # logging.info(images.size())
                    outputs = net(samples.to(device))
                    # if outputs.shape != labels.shape:
                    #    labels = torch.unsqueeze(labels, dim=-1).to(device).to(torch.float32)
                    val_loss.append(criterion(outputs.cpu(), labels))
                    y_true.append(labels)
                    y_prob_logit.append(outputs.data.cpu())
                val_loss = np.mean(val_loss)

                if config.is_regression:
                    loss = calc_loss(val_loss, "Valid", epoch)
                    if not (args.use_dp or config.use_dp) or (
                            (args.use_dp or config.use_dp) and is_epsilon_within_target(
                            target_epsilon=config.dp_target_epsilon,
                            actual_epsilon=privacy_engine.get_epsilon(config.dp_target_delta))):
                        # if we use DP, then we want the epsilon to be at least a specific value (e.g. 5% within target) to accept a DP model as best model.
                        if loss < best_loss:
                            best_loss = loss
                            best_epoch = epoch
                            torch.save(net.state_dict(), config.out_checkpoint_path)
                            if args.use_dp or config.use_dp:  # for ease of use, we want to allow activating DP both via args as well as via config. Both have default=False
                                epsilon = privacy_engine.get_epsilon(config.dp_target_delta)
                                logging.info(
                                    f"Saving best model so far at epoch {epoch} with DP(ε={epsilon},δ={config.dp_target_delta}) with loss = {loss} (best: lowest val_loss)"
                                )
                            else:
                                logging.info(
                                    f"Saving best model so far at epoch {epoch} with loss = {loss} (best: lowest val_loss)"
                                )
                else:  # classification
                    y_true = torch.cat(y_true)
                    y_prob_logit = torch.cat(y_prob_logit)
                    y_prob = from_logits_to_probs(y_prob_logit)
                    config_path = f"{args.config_path} with DP(ε={privacy_engine.get_epsilon(config.dp_target_delta)}, δ={config.dp_target_delta}) with target ε={config.dp_target_epsilon} and max_grad_norm={config.dp_max_grad_norm}" if (
                                args.use_dp or config.use_dp) else f"{args.config_path} without DP"
                    _, _, prec_rec_f1, roc_auc, prc_auc = calc_all_scores(
                        y_true,
                        y_prob,
                        val_loss,
                        "Valid",
                        config,
                        epoch,
                        config_path=config_path,
                        write_results_txt=False,
                        test_only=args.only_get_metrics,
                    )
                    if not (args.use_dp or config.use_dp) or (
                            (args.use_dp or config.use_dp) and is_epsilon_within_target(
                            target_epsilon=config.dp_target_epsilon,
                            actual_epsilon=privacy_engine.get_epsilon(config.dp_target_delta))):
                        # if we use DP, then we want the epsilon to be at least a specific value (e.g. 5% within target) to accept a DP model as best model.
                        val_f1 = prec_rec_f1[-1:][0]
                        # if val_loss < best_loss:
                        # if val_f1 > best_f1:
                        if prc_auc is None or np.isnan(prc_auc):
                            prc_auc = best_prc_auc
                        if prc_auc > best_prc_auc or (prc_auc == best_prc_auc and val_loss < best_loss):
                            best_loss = val_loss
                            best_f1 = val_f1
                            best_prc_auc = prc_auc
                            best_epoch = epoch
                            torch.save(net.state_dict(), config.out_checkpoint_path)
                            if args.use_dp or config.use_dp:  # for ease of use, we want to allow activating DP both via args as well as via config. Both have default=False
                                final_epsilon = privacy_engine.get_epsilon(config.dp_target_delta)
                                logging.info(
                                    f"Saving best model so far at epoch {epoch} with DP(ε={final_epsilon},δ={config.dp_target_delta}), with f1 = {val_f1}, au prc = {prc_auc}, and val_loss = {val_loss}  (best: highest prc_auc on validation)"
                                )
                            else:
                                logging.info(
                                    f"Saving best model so far at epoch {epoch} with f1 = {val_f1}, au prc = {prc_auc}, and val_loss = {val_loss}  (best: highest prc_auc on validation)"
                                )

        logging.info("Finished Training")
        logging.info(f"Saved best model state dict to {config.out_checkpoint_path}")
        if args.use_dp or config.use_dp:  # for ease of use, we want to allow activating DP both via args as well as via config. Both have default=False
            logging.info(
                f"Best model was achieved after {best_epoch} epochs, with val loss = {best_loss} and DP(ε={final_epsilon},δ={config.dp_target_delta})."
            )
        else:
            logging.info(
                f"Best model was achieved after {best_epoch} epochs, with val loss = {best_loss}"
            )

    # TESTING
    test(config=config, args=args, net=net, device=device, test_dataloaders=test_dataloaders, logfile_path=logfile_path,
         best_epoch=None if args.only_get_metrics else best_epoch, criterion=criterion, val_dataloader=val_dataloader,
         val_dataset=val_dataset, train_dataloader=train_dataloader, train_dataset=train_dataset, )


def make_net_private(net, config, dataloader) -> nn.Module:
    if config.optimizer_type == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=config.clf_lr, momentum=0.9)  # lr=0.001, momentum=0.9)
    elif config.optimizer_type == "adam":
        optimizer = optim.Adam(net.parameters(), lr=config.clf_lr, weight_decay=config.clf_weight_decay)
    elif config.optimizer_type == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr=config.clf_lr, weight_decay=config.clf_weight_decay)
    net, _, _, _ = make_private_with_epsilon(net=net,
                                             optimizer=optimizer,
                                             dataloader=dataloader,
                                             dp_target_epsilon=config.dp_target_epsilon,
                                             dp_target_delta=config.dp_target_delta,
                                             dp_max_grad_norm=config.dp_max_grad_norm,
                                             num_epochs=config.num_epochs,
                                             auto_validate_model=False,
                                             grad_sample_mode=None,  # "ew"
                                             )
    return net


def get_model_name(model_path) -> str:
    if 'swin_t_radimagenet_frozen' in model_path.lower():
        return 'swin_t_radimagenet_frozen'
    elif 'swin_t_imagenet_frozen' in model_path.lower():
        return 'swin_t_imagenet_frozen'
    elif 'swin_t_radimagenet' in model_path.lower():
        return 'swin_t_radimagenet'
    elif 'swin_t_imagenet' in model_path.lower():
        return 'swin_t_imagenet'
    elif 'swin_transformer' in model_path.lower():
        return 'swin_transformer'
    elif 'cnn' in model_path.lower():
        return 'cnn'
    else:
        #raise Exception(f"None of the available model names found in model path '{model_path}'")
        logging.warning(f"None of the available model names found in model path '{model_path}'. Returning 'swin_t_imagenet_frozen' as default.")
        return 'swin_t_imagenet_frozen'


def test(config, args, net, device, test_dataloaders: list, logfile_path, best_epoch, criterion, val_dataloader=None,
         val_dataset=None, train_dataloader=None, train_dataset=None):
    model_paths = []
    if args.only_get_metrics:
        model_paths = [str(in_path) for in_path in args.in_checkpoint_path if
                       len(str(in_path)) > 2]  # avoid commas introduced via cmd line
    else:
        model_paths.append(config.out_checkpoint_path)
    if len(model_paths) > 1:
        logging.info(f"Ensemble testing of models, as more than one model path detected: {model_paths}")
    models = prepare_models(config, args, model_paths, test_dataloaders, device, net)
    if val_dataloader is not None and args.validation_set:
        for i, model in enumerate(models):
            logging.info(f"Getting validation score for model {i + 1} of {len(models)}...")
            description = f'VALIDATION_seed{args.seed}_model{i + 1}of{len(models)}'
            get_val_scores(config, args, model, val_dataloader, device, description, write_results_txt=True)
    else:
        for idx, test_dataloader in enumerate(test_dataloaders):
            logging.info(f"Beginning test for test_dataloader {idx + 1} of {len(test_dataloaders)}...")
            y_probs_list = []
            for idx2, model in enumerate(models):
                logging.info(f"Beginning test for model {idx2 + 1} of {len(models)}...")
                with torch.no_grad():
                    y_true = []
                    y_prob_logit = []
                    test_loss = []
                    roi_type_arr = []
                    id_arr = []
                    logging.info(f"Testing for args: {args}")
                    for i, data in enumerate(tqdm(test_dataloader)):
                        try:
                            samples, labels, _, roi_types, ids, _ = data
                        except:
                            samples, labels, _, roi_types, ids = data
                        # logging.info(images.size())
                        samples, model = samples.to(device), model.to(device)
                        outputs = model(samples)
                        # if outputs.shape != labels.shape:
                        #    labels = torch.unsqueeze(labels, dim=-1).to(device).to(torch.float32)
                        test_loss.append(criterion(outputs.cpu(), labels))
                        y_true.append(labels)
                        y_prob_logit.append(outputs.data.cpu())
                        roi_type_arr.extend(roi_types)
                        id_arr.extend(ids)
                    # Accumulate results over entire dataset
                    test_loss = np.mean(test_loss)
                    y_true = torch.cat(y_true)
                    y_prob_logit = torch.cat(y_prob_logit)
                    y_prob = from_logits_to_probs(y_prob_logit)
                    y_probs_list.append(y_prob)
                # Calculate the metrics per model
                # write_results_txt = True if len(models)==1 else False # Do only write results to txt if we only test one model and not an ensemble?
                get_metrics_and_outputs(config, args, y_prob, y_true, test_loss, best_epoch, logfile_path,
                                        roi_type_arr, id_arr, split="test",
                                        description=f'seed{args.seed}_model{idx2 + 1}of{len(models)}_testdl{idx + 1}of{len(test_dataloaders)}',
                                        write_results_txt=True)
            test_ensemble(config, args, models, y_probs_list, idx, logfile_path, y_true, roi_type_arr, id_arr,
                          device=device, val_dataloader=val_dataloader, val_dataset=val_dataset,
                          test_dataloader=test_dataloader, train_dataloader=train_dataloader,
                          train_dataset=train_dataset, )
        logging.info("Finished testing.")


def get_val_scores(config, args, net, val_dataloader, device, description='get_val_scores', write_results_txt=False,
                   is_logged: bool = True):
    criterion = get_criterion(config)
    # VALIDATE
    net.to(device)
    val_loss = []
    with torch.no_grad():
        y_true = []
        y_prob_logit = []
        iterable = enumerate(tqdm(val_dataloader)) if is_logged else enumerate(val_dataloader)
        for i, data in iterable:
            try:
                samples, labels, _, _, _, _ = data
            except:
                samples, labels, _, _, _ = data
            outputs = net(samples.to(device))
            # if outputs.shape != labels.shape:
            #    labels = torch.unsqueeze(labels, dim=-1).to(device).to(torch.float32)
            val_loss.append(criterion(outputs.cpu(), labels))
            y_true.append(labels)
            y_prob_logit.append(outputs.data.cpu())
        y_true = torch.cat(y_true)
        y_prob_logit = torch.cat(y_prob_logit)
        y_prob = from_logits_to_probs(y_prob_logit)
        val_loss, acc_score, prec_rec_f1, roc_auc, prc_auc = calc_all_scores(
            y_true,
            y_prob,
            val_loss,
            "Valid",
            config,
            epoch=None,
            config_path=args.config_path,
            write_results_txt=write_results_txt,
            description=description,
            test_only=args.only_get_metrics,
            is_logged=is_logged,
        )
        return val_loss, acc_score, prec_rec_f1, roc_auc, prc_auc


def test_ensemble(config, args, models, y_probs_list, idx, logfile_path, y_true, roi_type_arr, id_arr, device,
                  val_dataloader=None, val_dataset=None, test_dataloader=None, train_dataloader=None,
                  train_dataset=None, ):
    # Model Ensembling
    logging.debug(f"models={models}")
    logging.info(f"Number of models: {len(models)}, Number of y_probs: {len(y_probs_list)}")
    if len(models) > 1:
        # 1 = voting by mean probabilities of all models i.e. mean over probabilities of multiple models
        if str(1) in args.ensemble_strategies:
            mean_y_probs = None
            stacked_y_probs = torch.stack(y_probs_list)
            logging.info(f"final stacked_y_probs.shape: {stacked_y_probs.shape}")
            logging.debug(f"final stacked_y_probs: {stacked_y_probs}")

            mean_y_probs = torch.mean(stacked_y_probs, 0)

            logging.info(f"final mean_y_probs.shape: {mean_y_probs.shape}")
            logging.debug(f"final mean_y_probs: {mean_y_probs}")
            # y_true, roi_type_arr, id_arr, and logfile_path don't change between models
            get_metrics_and_outputs(config=config, args=args, y_prob=mean_y_probs, y_true=y_true, test_loss=None,
                                    best_epoch=f"ensemble_w_strategy{args.ensemble_strategies}",
                                    logfile_path=logfile_path,
                                    roi_type_arr=roi_type_arr, id_arr=id_arr, split="test",
                                    description=f'ensemble_w_strategy_1of{args.ensemble_strategies}_of_{len(models)}_models_testdl{idx + 1}')
        if str(2) in args.ensemble_strategies:
            weight_multiplier_list = []
            mean_y_probs = None
            for i, y_probs in enumerate(y_probs_list):
                # get weight multiplier
                val_loss, val_acc_score, val_prec_rec_f1_scores, val_roc_auc, val_prc_auc = get_val_scores(config, args,
                                                                                                           net=models[
                                                                                                               i],
                                                                                                           val_dataloader=val_dataloader,
                                                                                                           device=device)
                weight_multiplier = get_weight_multiplier(args, val_loss, val_acc_score, val_prec_rec_f1_scores,
                                                          val_roc_auc, val_prc_auc)
                logging.info(
                    f"weight_multiplier {args.ensemble_weight_multiplier} of model {i + 1} of {len(y_probs_list)}: {weight_multiplier}")
                if args.weighting_by_rank:
                    weight_multiplier_list.append({'model_id': i, 'wm': weight_multiplier})
                else:
                    weight_multiplier_list.append(weight_multiplier)
                    y_probs = y_probs * weight_multiplier  # multiplying tensor with weight
                    logging.debug(f"{i} y_probs * weight_multiplier shape: {y_probs.shape}")
                    if mean_y_probs is None:
                        mean_y_probs = torch.zeros(y_probs.shape)
                    mean_y_probs = mean_y_probs.add(y_probs)  # addition of the two tensors
            if args.weighting_by_rank:
                # sort by wm value in ascending order (e.g. smallest value, i.e. worst model, first)
                sorted_weight_multiplier_list = sorted(weight_multiplier_list, key=lambda wml: wml['wm'])  #
                # now replace wm values with values based on rank of model
                weight_multiplier_list = []
                for i, swm in enumerate(sorted_weight_multiplier_list):
                    # replacing wm value by linearly scaling rank number
                    rank_based_wm = (i + 1) * (
                                i + 1)  # *10 # sorted_weight_multiplier_list is sorted in ascending order
                    weight_multiplier_list.append(rank_based_wm)
                    model_id = swm['model_id']
                    logging.info(f'model rank: {i}, model_id: {model_id}, weight:{rank_based_wm}')
                    y_probs = y_probs_list[model_id] * rank_based_wm
                    if mean_y_probs is None:
                        mean_y_probs = torch.zeros(y_probs.shape)
                    mean_y_probs = mean_y_probs.add(y_probs)  # addition of the two tensors

            # mean_y_probs.add(y_probs, alpha=weight_multiplier)  # addition of the two tensors
            #        mean_y_probs = torch.stack(mean_y_probs, y_probs)
            #    logging.info(f"{idx3} mean_y_probs shape: {mean_y_probs.shape}")
            # logging.info(f"{idx3} mean_y_probs: {mean_y_probs}")
            mean_y_probs = torch.div(mean_y_probs, sum(float(wm) for wm in weight_multiplier_list))
            get_metrics_and_outputs(config=config, args=args, y_prob=mean_y_probs, y_true=y_true, test_loss=None,
                                    best_epoch=f"ensemble_w_strategy{args.ensemble_strategies}",
                                    logfile_path=logfile_path,
                                    roi_type_arr=roi_type_arr, id_arr=id_arr, split="test",
                                    description=f'ensemble_w_strategy_2of{args.ensemble_strategies}_of_{len(models)}_models_testdl{idx + 1}')
        if str(3) in args.ensemble_strategies:
            mean_y_vote = None
            for i, y_probs in enumerate(y_probs_list):
                votes = torch.where(y_probs > 0.5, 1., 0.)
                if mean_y_vote is None:
                    mean_y_vote = torch.zeros(votes.shape)
                mean_y_vote = mean_y_vote.add(votes)  # addition of the two tensors
                logging.info(f"mean_y_vote in iteration={i}: {mean_y_vote}")

            logging.debug(f"final y_probs_list : {y_probs_list}")

            # stacked_y_probs = torch.stack(y_probs_list)
            # logging.info(f"final stacked_y_probs.shape: {stacked_y_probs.shape}")
            # votes = torch.where(stacked_y_probs > 0.5, 1., 0.)
            logging.debug(f"mean_y_vote before division by {len(y_probs_list)} shape : {mean_y_vote.shape}")
            logging.debug(f"mean_y_vote before division by {len(y_probs_list)}: {mean_y_vote}")
            # mean_y_vote = torch.mean(mean_y_probs, 0) # if >0.5 majority vote

            mean_y_vote = torch.div(mean_y_vote, len(y_probs_list))

            # y_true, roi_type_arr, id_arr, and logfile_path don't change between models
            get_metrics_and_outputs(config=config, args=args, y_prob=mean_y_vote, y_true=y_true, test_loss=None,
                                    best_epoch=f"ensemble_w_strategy{args.ensemble_strategies}",
                                    logfile_path=logfile_path,
                                    roi_type_arr=roi_type_arr, id_arr=id_arr, split="test",
                                    description=f'ensemble_w_strategy_3of{args.ensemble_strategies}_of_{len(models)}_models_testdl{idx + 1}')
        if str(4) in args.ensemble_strategies:
            weight_multiplier_list = []
            mean_y_vote = None
            for i, y_probs in enumerate(y_probs_list):
                votes = torch.where(y_probs > 0.5, 1., -1.)  # -1 instead of 0 needed for weighted voting
                logging.info(f"final votes shape : {votes.shape}")
                logging.debug(f"final votes : {votes}")
                val_loss, val_acc_score, val_prec_rec_f1_scores, val_roc_auc, val_prc_auc = get_val_scores(config, args,
                                                                                                           net=models[
                                                                                                               i],
                                                                                                           val_dataloader=val_dataloader,
                                                                                                           device=device)
                weight_multiplier = get_weight_multiplier(args, val_loss, val_acc_score, val_prec_rec_f1_scores,
                                                          val_roc_auc, val_prc_auc)
                weight_multiplier_list.append(weight_multiplier)
                logging.info(
                    f"weight_multiplier {args.ensemble_weight_multiplier} of model {i + 1} of {len(y_probs_list)}: {weight_multiplier}")
                votes = votes * weight_multiplier  # multiplying votes tensor with weight
                if mean_y_vote is None:
                    mean_y_vote = torch.zeros(votes.shape)
                mean_y_vote = mean_y_vote.add(votes)  # addition of the two tensors
            logging.info(f"mean_y_vote before division: {mean_y_vote} ")
            mean_y_vote = torch.div(mean_y_vote, sum(float(wm) for wm in
                                                     weight_multiplier_list))  # (weight_multiplier*len(y_probs_list)) # get the weighted mean by dividing probabilities by weight multipliers
            logging.info(
                f"mean_y_vote.shape after division by {sum(float(wm) for wm in weight_multiplier_list)}: {mean_y_vote.shape}")
            logging.info(
                f"mean_y_vote after division by {sum(float(wm) for wm in weight_multiplier_list)}: {mean_y_vote}")
            # outcomment (and change -1. above to 0.) if vote fraction should be treated as probability (e.g. 2/3 votes yes = 0.667 instead of 1 fed to auc calculation)
            mean_y_vote = torch.where(mean_y_vote > 0, 1.,
                                      0.)  # before we had a vote of either 1 or -1, so 0 is in the middle.
            logging.info(f"mean_y_vote after assigning 0 or 1 based on votes: {mean_y_vote}")
            logging.info(f"mean_y_vote after assigning 0 or 1 based on votes: {mean_y_vote}")

            # y_true, roi_type_arr, id_arr, and logfile_path don't change between models
            get_metrics_and_outputs(config=config, args=args, y_prob=mean_y_vote, y_true=y_true, test_loss=None,
                                    best_epoch=f"ensemble_w_strategy{args.ensemble_strategies}",
                                    logfile_path=logfile_path,
                                    roi_type_arr=roi_type_arr, id_arr=id_arr, split="test",
                                    description=f'ensemble_w_strategy_4of{args.ensemble_strategies}_of_{len(models)}_models_testdl{idx + 1}')

        if str(5) in args.ensemble_strategies:
            assert test_dataloader is not None, "You need to provide a test dataloader to run ensemble strategy 5 (nn search on val set metric based weighting)"
            assert val_dataset is not None, "You need to provide a val_dataset to run ensemble strategy 5 (nn search on val set metric based weighting)"
            if args.use_training_in_ann:
                initial_metadata = copy.deepcopy(train_dataset.metadata)
            else:
                initial_metadata = copy.deepcopy(val_dataset.metadata)
                # here we give weight to the models based on their validation subset performance.
            # we define a searchable index of embeddings (image representations in vector space)
            embedding_model = load_embedding_model(embedding_model_type=args.embedding_model_type, device=device,
                                                   config=config)
            logging.info(
                f"Using embedding model (type: {args.embedding_model_type}, class: {embedding_model.__class__.__name__} ) on {device}..")
            if args.use_training_in_ann:
                embedding_index = get_index_from_dataloader(my_dataloader=train_dataloader, model=embedding_model,
                                                            embedding_model_type=args.embedding_model_type,
                                                            distance_metric='angular', num_trees=10, filename='val.ann',
                                                            device=device)
            else:
                embedding_index = get_index_from_dataloader(my_dataloader=val_dataloader, model=embedding_model,
                                                            embedding_model_type=args.embedding_model_type,
                                                            distance_metric='angular', num_trees=10, filename='val.ann',
                                                            device=device)
            # now, we need to get the top k most similar embeddings for each image in the test dataset
            # then we get a validation dataloader only containing these k most similar image embeddings
            # with this top k validation dataloader, we test the validation performance for each model and store the results
            mean_y_probs = torch.zeros(y_probs_list[0].shape)
            model_wm_tracker = []  # tracking models at the end
            counter = 0
            for i, data in enumerate(tqdm(test_dataloader)):
                try:
                    sample, labels, _, roi_types, ids, _, j = data
                except:
                    sample, labels, _, roi_types, ids, j = data
                # j is the index of the sample in the dataloader. batch size should already be set to 1 here.
                top_k = get_top_k_from_image(image=sample.to(device), model=embedding_model,
                                             embedding_index=embedding_index,
                                             embedding_model_type=args.embedding_model_type, n=int(args.top_k))
                if args.use_training_in_ann:
                    nn_dataloader = get_nn_dataloader(initial_dataset=train_dataset, nn_indices=top_k,
                                                      batch_size=config.batch_size, initial_dataloader=train_dataloader,
                                                      initial_metadata=initial_metadata)
                else:
                    nn_dataloader = get_nn_dataloader(initial_dataset=val_dataset, nn_indices=top_k,
                                                      batch_size=config.batch_size, initial_dataloader=val_dataloader,
                                                      initial_metadata=initial_metadata)
                weight_multiplier_list = []
                val_loss_list = []
                y_probs_sample = 0

                for k, model in enumerate(models):
                    # We want to get the weights for each model. To do so, we test each model on a test case nearest neighbors from the validation set
                    val_loss, val_acc_score, val_prec_rec_f1_scores, val_roc_auc, val_prc_auc = get_val_scores(config,
                                                                                                               args,
                                                                                                               net=model,
                                                                                                               val_dataloader=nn_dataloader,
                                                                                                               device=device,
                                                                                                               is_logged=False)
                    weight_multiplier = get_weight_multiplier(args, val_loss, val_acc_score, val_prec_rec_f1_scores,
                                                              val_roc_auc, val_prc_auc)
                    weight_multiplier_list.append(weight_multiplier)
                    # multiplying model's probability by its weight multiplier and summing the probabilites over all models in y_probs_sample
                    if args.remove_all_but_l_models is None:  # TODO
                        y_probs_sample = y_probs_sample + (y_probs_list[k][
                                                               j] * weight_multiplier)  # (y_probs_list[k][((len(sample)-1) * i) + j] * weight_multiplier) # getting the probabilities for the right test sample (taking batch size into account) from the right model k
                        logging.info(
                            f"Model: {k}, test sample: {j}, val_loss: {val_loss}, val_prc_auc: {val_prc_auc}, wm: {weight_multiplier}.")
                    else:
                        val_loss_list.append(val_loss)
                if len(model_wm_tracker) == 0:
                    model_wm_tracker = [0 for x in weight_multiplier_list]  # init if need be
                # summing all weight multipliers (To find best models overall)
                model_wm_tracker = [sum(x) for x in zip(weight_multiplier_list, model_wm_tracker)]
                # Option: Remove a number of l models from ensemble prediction for this sample based on weight multiplier values
                if args.remove_all_but_l_models is not None:
                    # get top l ("num_models") form the weight_multipliers
                    wm_array = np.array(weight_multiplier_list)  # transform to numpy
                    remaining_wms = np.argsort(-wm_array)[:int(
                        args.remove_all_but_l_models)].tolist()  # get indices of highest weight multiplier
                    for k, model in enumerate(models):
                        if k in remaining_wms:
                            # only in this case we add to the y_probs_sample
                            if "b" in args.ensemble_strategies:
                                # we set weight multiplier to 1 (which means, we are not weighting at all and the ANN exercise is just to find the best models
                                weight_multiplier_list[k] = 1.
                            y_probs_sample = y_probs_sample + (y_probs_list[k][j] * weight_multiplier_list[k])
                        else:
                            weight_multiplier_list[k] = 0.
                        logging.info(
                            f"5b, model: {k}, test sample: {j}, val_loss: {val_loss_list[k]}, wm: {weight_multiplier_list[k]}.")

                # TODO: If need be, here, we could further implement a ranking based weighting as in ensemble strategy 2 (args.weighting_by_rank)
                # dividing by the sum of weight multipliers to get weighted average over models per sample
                mean_y_probs_sample = torch.div(y_probs_sample, sum(float(wm) for wm in weight_multiplier_list))
                mean_y_probs[j] = mean_y_probs_sample  # writing probabilities into corresponding index in tensor
                # if i >= 20: break # Testing without waiting too much ;)

                if args.num_models_to_remove is not None and counter == config.batch_size and len(models) > int(
                        args.min_num_models):
                    epoch_dict = {}  # tracking models each epoch
                    mean_tracked_wm = sum(float(wm) for wm in model_wm_tracker) / len(model_wm_tracker)
                    for idx, model_path in enumerate(models):
                        epoch_dict[idx] = model_wm_tracker[idx]
                    # sorting model with the smallest
                    sorted_epoch_dict = sorted(epoch_dict.items(), key=lambda x: x[1])  # reverse=True)
                    # get idx of j lowest ranking models
                    removable_model_idx = []
                    for j in range(int(args.num_models_to_remove)):
                        if len(models) > int(args.min_num_models):
                            model_wm_score = sorted_epoch_dict[j]
                            for id, score in epoch_dict.items():
                                logging.info(f"score={score}, index={id}, search_for_score={model_wm_score}")
                                if score == model_wm_score[1] and model_wm_score[0] not in removable_model_idx:
                                    model_idx = model_wm_score[0]
                                    break  # break inner loop
                            logging.info(
                                f"Now getting model {model_idx} from models 0-{len(args.in_checkpoint_path) - 1}: {args.in_checkpoint_path[model_idx]}")
                            removable_model_idx.append(model_idx)
                    logging.info(
                        f"removable_model_idx: {removable_model_idx}, max(removable_model_idx): {max(removable_model_idx)}")
                    while len(removable_model_idx) > 0:
                        highest_model_index = max(removable_model_idx)  # high to low to avoid shifting indices
                        removable_model_idx.remove(highest_model_index)
                        model_path = args.in_checkpoint_path[highest_model_index]
                        logging.info(
                            f"Removing {args.num_models_to_remove} models. Now removing model with index {highest_model_index}, as it had a wm score of {model_wm_tracker[highest_model_index]} (compared to mean wm across all {len(args.in_checkpoint_path)} models: {mean_tracked_wm}). Removed model path: {model_path}")
                        del models[highest_model_index]
                        del args.in_checkpoint_path[highest_model_index]
                        del y_probs_list[
                            highest_model_index]  # FIXME Check the following: model_idx does refer to adjusted args.in_checkpoint_patch - but index should be the same as we equally adjust y_probs_list
                        del model_wm_tracker[highest_model_index]
                counter = 0 if counter == config.batch_size else counter + 1
            model_wm_tracker_dict = {}
            for idx, model_path in enumerate(args.in_checkpoint_path):
                model_wm_tracker_dict[model_path] = model_wm_tracker[idx]
            sorted_model_wm_tracker_dict = sorted(model_wm_tracker_dict.items(), key=lambda x: x[1], reverse=True)
            try:
                file_path = Path('sorted_model_wm_tracker_dict.txt')
                if file_path.is_file():
                    with open(file_path, 'a') as f:
                        f.write(f"{sorted_model_wm_tracker_dict}")
            except Exception as e:
                logging.warning(f"Failure while trying to write to sorted_model_wm_tracker_dict.txt: {e}")
            logging.info(f"sorted_model_wm_tracker_dict: {sorted_model_wm_tracker_dict}")
            get_metrics_and_outputs(config=config, args=args, y_prob=mean_y_probs, y_true=y_true, test_loss=None,
                                    best_epoch=f"ensemble_w_strategy{args.ensemble_strategies}",
                                    logfile_path=logfile_path,
                                    roi_type_arr=roi_type_arr, id_arr=id_arr, split="test",
                                    description=f'ensemble_w_strategy_5of{args.ensemble_strategies}_of_{len(models)}_models_testdl{idx + 1}')


def get_weight_multiplier(args, val_loss, val_acc_score, val_prec_rec_f1_scores, val_roc_auc, val_prc_auc):
    if args.ensemble_weight_multiplier == "val_loss":
        return 1. / val_loss  # divison as val_loss can be >1 TODO: Recalculate values based on this important change # 1. - val_loss
    if args.ensemble_weight_multiplier == "val_loss^2":
        return 1. / (
                    val_loss ** 2)  # divison as val_loss can be >1 TODO: Recalculate values based on this important change # 1. - val_loss
    elif args.ensemble_weight_multiplier == "val_acc":
        return val_acc_score
    elif args.ensemble_weight_multiplier == "val_auroc":
        return val_roc_auc
    elif args.ensemble_weight_multiplier == "val_auprc":
        return val_prc_auc
    elif args.ensemble_weight_multiplier == "val_auprc^3":
        return val_prc_auc ** 3
    else:
        raise NotImplementedError(
            f"No weight multiplier found for {args.ensemble_weight_multiplier}. Please try different value, one of: val_loss, val_loss^2, val_acc, val_auroc, val_auprc")


def prepare_models(config, args, model_paths, test_dataloaders, device, net):
    models = []
    net.train()  # seems to be needed to avoid torch IllegalModuleConfigurationError
    for model_path in model_paths:
        if not os.path.exists(model_path):
            raise Exception(f"No checkpoint found in '{model_path}'. Revise and restart test.")
        model_name = get_model_name(model_path=model_path)
        logging.info(f"Loading model {model_name} from {model_path}")
        config.model_name = model_name
        # try:
        if (
                args.use_dp or config.use_dp):  # for ease of use, we want to allow activating DP both via args as well as via config. Both have default=False
            # All of the below if condition is solely to adjust the net as if it would have been DP-trained to allow for loading DP weights without errors.
            net = make_net_private(net, config, test_dataloaders[0])
        if not args.only_get_metrics:
            # net has just been trained and is already initialised
            net.load_state_dict(torch.load(model_path, map_location=device))
        else:
            # However, if only_get_metrics is true, no net has been training, we just want to test a predefined model
            if "_e" in model_path:  # _e indicates a private model
                # in private model mode, get net but without directly loading weights.
                net = get_classifier(config=config, device=device)
                # load dp adjusted weights
                net = make_net_private(net, config, test_dataloaders[0])
                net.load_state_dict(torch.load(model_path, map_location=device))
            else:
                # in non-private model, get net with loaded trained weights.
                net = get_classifier(config=config, weights_path=model_path, device=device)
        net.eval()
        models.append(net)
        # except Exception as e:
        #    raise Exception(f"Error while trying to load model from path '{model_path}': {e}")
    return models


def get_metrics_and_outputs(config, args, y_prob, y_true, test_loss, best_epoch, logfile_path, roi_type_arr, id_arr,
                            split="test", description='', write_results_txt=True):
    loss, acc_score, prec_rec_f1_scores, roc_auc, prc_auc = calc_all_scores(y_true, y_prob, test_loss, split, config,
                                                                            epoch=best_epoch,
                                                                            config_path=args.config_path,
                                                                            write_results_txt=write_results_txt,
                                                                            description=description,
                                                                            test_only=args.only_get_metrics, )  # using y_prob instead of the logit.
    output_per_sample_results(config, id_arr, y_prob, logfile_path, description=description)
    # Get metrics for specific ROI types, i.e. masses and calcifications, separately
    logging.warning(f"ROI types in test dataset: {set(roi_type_arr)}")
    get_metric_per_roi_type(type="mass", config=config, roi_type_arr=roi_type_arr, y_true=y_true,
                            y_prob=y_prob, split=split)
    get_metric_per_roi_type(type="calcification", config=config, roi_type_arr=roi_type_arr, y_true=y_true,
                            y_prob=y_prob, split=split)
    if config.binary_classification:
        output_ROC_curve(y_true, y_prob, split, config, logfile_path, description=description)
    return loss, acc_score, prec_rec_f1_scores, roc_auc, prc_auc


def is_epsilon_within_target(target_epsilon, actual_epsilon, allowed_derivation_upper_bound=0.05,
                             allowed_derivation_lower_bound=0.25):  # previously allowed derivation until 26.01.2022 = 0.05
    """ we allow the epsilon of a model to be inside a range to accept a model. """
    upper_bound = target_epsilon + target_epsilon * allowed_derivation_upper_bound
    lower_bound = target_epsilon - target_epsilon * allowed_derivation_lower_bound
    is_within = actual_epsilon < upper_bound and actual_epsilon > lower_bound
    logging.info(
        f"Is epsilon (actual:{actual_epsilon})  within target ({target_epsilon})? Result: {is_within}. Lower bound: {lower_bound}, upper bound: {upper_bound}")
    return is_within


def save_entire_dataset(net, dataset):
    # This code section is only for saving patches as image files and further info about the patch if needed.
    # The program stops execution after this section and performs no training.

    logging.info(f"Saving data samples...")

    # TODO: state_dict name should probably be in config yaml instead of hardcoded.
    net.load_state_dict(
        torch.load("model_checkpoints/classifier 50 no synth/classifier.pt")
    )
    net.eval()
    # TODO refactor (make optional & parametrize) or remove saving dataset
    save_data_path = Path("save_dataset")

    with open(save_data_path / "validation.txt", "w") as f:
        f.write("index, y_prob\n")
    cnt = 0
    metapoints = []
    for data in tqdm(
            dataset
    ):  # this has built-in shuffling; if not shuffled, only lesioned patches will be output first
        sample, label, image, r, d = data
        outputs = net(sample[np.newaxis, ...])

        # for y_prob_logit, label, image, r, d in zip(outputs.data, labels, images, rs, ds):
        y_prob_logit = outputs.data
        y_prob = from_logits_to_probs(y_prob_logit)  # torch.exp(y_prob_logit)
        metapoint = {
            "label": label,
            "roi_type": r,
            "dataset": d,
            "cnt": cnt,
            "y_prob": y_prob.numpy().tolist(),
        }
        metapoints.append(metapoint)

        # with open(save_data_path / "validation.txt", "a") as f:
        #     f.write(f'{cnt}, {y_prob}\n')

        label = "healthy" if int(label) == 1 else "with_lesions"
        out_image_dir = save_data_path / "validation" / str(label)
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_image_path = out_image_dir / f"{cnt}.png"
        cv2.imwrite(str(out_image_path), np.array(image))

        cnt += 1
    with open(save_data_path / "validation.json", "w") as f:
        f.write(json.dumps(metapoints))

    logging.info(f"Saved data samples to {save_data_path.resolve()}")


def get_label_balance_info(dataset, ds_name: str, config):
    if not config.binary_classification:
        logging.warning("You are not using binary CLF. This implementation will only work for binary classification. ")
    if config.classes == "is_healthy":
        num_negative, num_positive = dataset.len_of_classes()
    else:
        labels = [data_point[1] for data_point in dataset]
        num_negative = labels.count(0)  # e.g. malignant
        num_positive = labels.count(1)  # e.g. benign

    logging.info(f"{ds_name} dataset:")
    logging.info(f"{ds_name} -> Negative: {num_negative}, Positive: {num_positive}")
    logging.info(f"{ds_name} -> Share of positives: {num_positive / len(dataset)}")
    return num_negative, num_positive


def arrange_weights(weight_class_positive, weight_class_negative, dataset):
    return [
        weight_class_negative if data_point[1] == 0 else weight_class_positive
        for i, data_point in enumerate(dataset)
    ]


def output_per_sample_results(config, id_arr, y_prob, logfile_path, description=''):
    if config.output_classification_result in {"json", "csv"}:
        df_meta = pd.read_json(config.metadata_path)
        df_results = pd.DataFrame(
            {
                "patch_id": id_arr,
                f"y_prob": np.array(y_prob[:, -1]),
            }
        )
        df_results["patch_id"] = df_results["patch_id"].astype("int64")
        df_meta = pd.merge(df_meta, df_results, how="inner", on="patch_id")
        if config.output_classification_result == "json":
            df_meta.to_json(
                f"{logfile_path.replace('.txt', f'_{description}')}.json", orient="records"
            )
        else:
            df_meta.to_csv(
                f"{logfile_path.replace('.txt', f'_{description}')}.csv", index=False
            )


def get_criterion(config):
    if config.is_regression:
        return nn.MSELoss()  # nn.L1Loss()
    else:
        return nn.CrossEntropyLoss(label_smoothing=config.clf_label_smoothing)
        # criterion = nn.BCEWithLogitsLoss(label_smoothing=config.clf_label_smoothing)


def get_metric_per_roi_type(type: str, config, roi_type_arr, y_true, y_prob, split="test"):
    if type in config.data[split].roi_types:
        indices = [
            i
            for i, item in enumerate(roi_type_arr)
            if item == type or item == "healthy"]
        calc_AUROC(
            y_true[indices], y_prob[indices], f"Test only: {type}", config
        )
        calc_AUPRC(
            y_true[indices], y_prob[indices], f"Test only: {type}", config
        )


if __name__ == "__main__":
    main()
