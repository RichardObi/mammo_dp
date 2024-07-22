import logging
from pathlib import Path
import os

try:
    import tkinter
except:
    # Need to use matplotlib without tkinter dependency
    # tkinter is n.a. in some python distributions
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def from_logits_to_probs(logits):
    """
    Turns logits into probabilities along dim=1. Returns the probabilities for all classes.
    """
    return F.softmax(logits, dim=1)


def calc_all_scores(y_true, y_prob, v_loss, run_type, config, epoch=None, config_path=None, write_results_txt=True, description='', test_only=False, is_logged=True):
    """Calculates, outputs, and returns loss, accuracy, precision, recall, F1-score, ROC_AUC, and PRC_AUC

    Args:
        y_true ([type]): labels for each sample and each class (N x C)
        y_prob ([type]): probabilities for each sample and each class (N x C)
        v_loss ([type]): loss of each sample (N)
        run_type (String): either 'train', 'val', or 'test'
        epoch (int, optional): which epoch number to print. Defaults to None.

    Returns:
        [type]: [description]
    """
    y_pred = torch.argmax(y_prob, 1)  # class prediction for each sample
    if v_loss is not None and not (isinstance(v_loss, list) and len(v_loss) == 0):
        loss = calc_loss(v_loss, run_type, epoch, is_logged)
    else:
        loss = v_loss
    acc_score = calc_accuracy(y_true, y_pred, run_type, config, epoch, is_logged)
    prec_rec_f1_scores = calc_prec_rec_f1_scores(
        y_true, y_pred, run_type, config, epoch, is_logged
    )
    prec, rec, f1 = prec_rec_f1_scores
    # TODO calculating prec, rec, F1, AUROC, AURPC may be only applicable if classifying binary here due to ?
    roc_auc = calc_AUROC(y_true, y_prob, run_type, config, epoch, is_logged)
    prc_auc = calc_AUPRC(y_true, y_prob, run_type, config, epoch, is_logged)
    if run_type.lower() == "test" or (run_type.lower() == "valid" and write_results_txt):
        logging.info(f"{config_path} {description}")
        logging.info(f"{epoch}\t{loss}\t{acc_score}\t{prec}\t{rec}\t{f1}\t{roc_auc}\t{prc_auc}")
        if write_results_txt:
            file_path = Path('results.txt')
            if test_only:
                file_path = Path('results_test.txt')
            if file_path.is_file():
                with open(file_path, 'a') as f:
                    f.write(f"\n{config_path} {description}\t{epoch}\t{loss}\t{acc_score}\t{prec}\t{rec}\t{f1}\t{roc_auc}\t{prc_auc}")
    return loss, acc_score, prec_rec_f1_scores, roc_auc, prc_auc


def calc_loss(v_loss, run_type, epoch=None, is_logged=True):
    loss = np.mean(v_loss)
    if epoch is None and is_logged:
        logging.info(f"{run_type} loss: {loss}")
    elif is_logged:
        logging.info(f"{run_type} loss in {epoch} epoch: {loss}")
    return loss


def calc_accuracy(y_true, y_pred, run_type, config, epoch=None, is_logged=True):
    if config.num_classes < 2:
        logging.warn("Accuracy is only defined for classification tasks.")
        return None

    score = accuracy_score(y_true, y_pred)
    if epoch is None and is_logged:
        logging.info(f"{run_type} accuracy:  {score}")
    elif is_logged:
        logging.info(f"{run_type} accuracy in {epoch} epoch:  {score}")
    return score


def calc_prec_rec_f1_scores(y_true, y_pred, run_type, config, epoch=None, is_logged=True):
    if config.num_classes < 2:
        logging.warn("Prec, rec, and F1 are only defined for classification tasks.")
        return None

    prec, rec, f1, _ = np.array(precision_recall_fscore_support(y_true, y_pred))[
        :, -1
    ]  # keep scores for label==1

    epoch_text = "" if epoch is None else f" in {epoch} epoch"
    if is_logged:
        logging.info(f"{run_type} Precision{epoch_text}: {prec}")
        logging.info(f"{run_type} Recall{epoch_text}:    {rec}")
        logging.info(f"{run_type} F1-Score{epoch_text}:  {f1}")

    return prec, rec, f1


def calc_AUROC(y_true, y_prob, run_type, config, epoch=None, is_logged=True):
    if config.num_classes != 2:
        logging.warning("AUROC currently only works for binary classification.")
        return None

    try:
        roc_auc = roc_auc_score(y_true, y_prob[:, -1])
        if epoch is None and is_logged:
            logging.info(f"{run_type} AUROC: {roc_auc}")
        elif is_logged:
            logging.info(f"{run_type} AUROC in {epoch} epoch: {roc_auc}")
        return roc_auc
    except ValueError as e:
        logging.warning(
            f"WARNING cannot calculate AUROC. calc_AUROC got a ValueError. This happens for example when only one class is present in the dataset; run_type: {run_type}; epoch: {epoch}: {e}"
        )
    except IndexError as e:
        logging.warning(
            f"WARNING cannot calculate AUROC. calc_AUROC got an IndexError. This happens for example when the respective dataset is empty, for example when filtering for calcifications although there are no calcifications; run_type: {run_type}; epoch: {epoch}: {e}"
        )


def calc_AUPRC(y_true, y_prob, run_type, config, epoch=None, is_logged=True):
    if config.num_classes != 2:
        logging.warn("AUPRC currently only works for binary classification.")
        return None

    try:
        prc_auc = average_precision_score(y_true, y_prob[:, -1])
        if epoch is None and is_logged:
            logging.info(f"{run_type} AUPRC: {prc_auc}")
        elif is_logged:
            logging.info(f"{run_type} AUPRC in {epoch} epoch: {prc_auc}")
        return prc_auc
    except ValueError as e:
        logging.warning(
            f"WARNING cannot calculate AUPRC. calc_AUPRC got a ValueError. This happens for example when only one class is present in the dataset;; run_type: {run_type}; epoch: {epoch}: {e}"
        )
    except IndexError as e:
        logging.warning(
            f"WARNING cannot calculate AUPRC. calc_AUPRC got an IndexError. This happens for example when the respective dataset is empty, for example when filtering for calcifications although there are no calcifications; run_type: {run_type}; epoch: {epoch}: {e}"
        )



def output_ROC_curve(y_true, y_prob_logit, run_type, config, logfile_path = None, description=''):

    """Only for binary classification

    Args:
        y_true ([type]): [description]
        y_prob ([type]): [description]
        run_type ([type]): [description]
        logfilename ([type]): location where to save the output figure
    """

    if config.num_classes != 2:
        logging.warning("Can only output ROC curve for binary classification.")

        return None

    y_prob = torch.exp(y_prob_logit)[
        :, -1
    ]  # keep only the probabilities for the true class

    # Calculate ROC values:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC values:
    plt.figure(figsize=(18, 18))
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{run_type} set; receiver operating characteristic")
    plt.legend(loc="lower right")
    if logfile_path is None:
        logfile_path = ""
    plt.savefig(str(Path(logfile_path)).replace('.txt', '').replace('.log', '') + f"_ROC_curve_{description}.png")
