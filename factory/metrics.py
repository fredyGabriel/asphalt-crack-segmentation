import torch
from typing import Tuple


def confusion_matrix(prediction: torch.Tensor, target: torch.Tensor) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate confusion matrix elements for binary segmentation.

    Parameters:
        prediction: Binary segmentation prediction (1 for positive, 0 for
            negative)
        target: Binary ground truth (1 for positive, 0 for negative)

    Returns:
        tuple: (TP, FP, TN, FN) where:
            TP: True Positives
            FP: False Positives
            TN: True Negatives
            FN: False Negatives
    """
    true_positive = (prediction * target).sum().float()
    false_positive = prediction.sum().float() - true_positive
    false_negative = target.sum().float() - true_positive
    true_negative = target.numel() - true_positive - false_positive -\
        false_negative

    return true_positive, false_positive, true_negative, false_negative


def iou(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6)\
        -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU), also known as Jaccard index.

    IoU measures the overlap between the predicted segmentation and the ground
        truth.
    Formula: IoU = TP / (TP + FP + FN)

    Parameters:
        prediction: Binary segmentation prediction (1 for positive, 0 for
            negative)
        target: Binary ground truth (1 for positive, 0 for negative)
        smooth: Small constant to avoid division by zero

    Returns:
        IoU score between 0 and 1
    """
    tp, fp, tn, fn = confusion_matrix(prediction, target)
    return (tp + smooth) / (tp + fp + fn + smooth)


def pixel_accuracy(prediction: torch.Tensor, target: torch.Tensor) ->\
        torch.Tensor:
    """
    Calculate pixel accuracy for segmentation.

    The pixel accuracy represents the percentage of pixels correctly
        classified.
    Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Parameters:
        prediction: Binary segmentation prediction (1 for positive, 0 for
            negative)
        target: Binary ground truth (1 for positive, 0 for negative)

    Returns:
        Pixel accuracy between 0 and 1
    """
    tp, fp, tn, fn = confusion_matrix(prediction, target)
    return (tp + tn) / (tp + fp + tn + fn)


def dice_coefficient(prediction: torch.Tensor, target: torch.Tensor,
                     smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Dice coefficient, also known as F1 score.

    The Dice coefficient measures the similarity between the predicted
        segmentation and the ground truth.
    Formula: Dice = 2 * TP / (2 * TP + FP + FN)

    Parameters:
        prediction: Binary segmentation prediction (1 for positive, 0 for
            negative)
        target: Binary ground truth (1 for positive, 0 for negative)
        smooth: Small constant to avoid division by zero

    Returns:
        Dice coefficient between 0 and 1
    """
    tp, fp, tn, fn = confusion_matrix(prediction, target)
    return (2 * tp + smooth) / (2 * tp + fp + fn + smooth)


def precision(prediction: torch.Tensor, target: torch.Tensor,
              smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate precision: TP / (TP + FP)

    Parameters:
        prediction: Binary segmentation prediction (1 for positive, 0 for
            negative)
        target: Binary ground truth (1 for positive, 0 for negative)
        smooth: Small constant to avoid division by zero

    Returns:
        Precision score between 0 and 1
    """
    tp, fp, tn, fn = confusion_matrix(prediction, target)
    return (tp + smooth) / (tp + fp + smooth)


def recall(prediction: torch.Tensor, target: torch.Tensor,
           smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate recall (sensitivity): TP / (TP + FN)

    Parameters:
        prediction: Binary segmentation prediction (1 for positive, 0 for
            negative)
        target: Binary ground truth (1 for positive, 0 for negative)
        smooth: Small constant to avoid division by zero

    Returns:
        Recall score between 0 and 1
    """
    tp, fp, tn, fn = confusion_matrix(prediction, target)
    return (tp + smooth) / (tp + fn + smooth)


def f1_score(prediction: torch.Tensor, target: torch.Tensor,
             smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate F1 score: 2 * (precision * recall) / (precision + recall)
    This is equivalent to the Dice coefficient but expressed in terms of
        precision and recall

    Parameters:
        prediction: Binary segmentation prediction (1 for positive, 0 for
            negative)
        target: Binary ground truth (1 for positive, 0 for negative)
        smooth: Small constant to avoid division by zero

    Returns:
        F1 score between 0 and 1
    """
    tp, fp, tn, fn = confusion_matrix(prediction, target)
    prec = (tp + smooth) / (tp + fp + smooth)
    rec = (tp + smooth) / (tp + fn + smooth)
    return 2 * (prec * rec) / (prec + rec + smooth)
