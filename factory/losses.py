import torch
import torch.nn as nn


class BinaryCrossEntropyLoss(nn.Module):
    """
    Binary Cross Entropy Loss for binary segmentation tasks.

    This class is a wrapper around PyTorch's BCEWithLogitsLoss, which combines
    a Sigmoid layer and the BCELoss in one single class for improved numerical
    stability compared to using a plain Sigmoid followed by BCELoss.

    Attributes:
        loss_fn (nn.BCEWithLogitsLoss): The binary cross entropy loss
            function with logits.

    Example:
        >>> criterion = BinaryCrossEntropyLoss()
        >>> output = model(input)  # shape: (B, 1, H, W) - raw logits
        >>> target = torch.ones_like(output)  # (B, 1, H, W) with values 0 or 1
        >>> loss = criterion(output, target)
        >>> loss.backward()
    """
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Calculate the Binary Cross Entropy Loss between predicted logits and
        targets.

        Args:
            inputs (torch.Tensor): The prediction logits (unbounded).
            targets (torch.Tensor): The target values (0 or 1).

        Returns:
            torch.Tensor: The computed binary cross entropy loss.
        """
        return self.loss_fn(inputs, targets)


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks.

    This class implements the Dice Loss, which is based on the Dice
        coefficient, a measure of overlap between two samples. It is commonly
        used for segmentation tasks where class imbalance is present.

    The formula for Dice Loss is:
    DiceLoss = 1 - (2 * intersection + smooth) / (sum_pred + sum_target +
        smooth)

    Where intersection = sum(pred * target)

    Attributes:
        smooth (float): Smoothing term to avoid division by zero
        sigmoid (bool): Whether to apply sigmoid activation to input logits

    Example:
        >>> criterion = DiceLoss()
        >>> output = model(input)  # shape: (B, 1, H, W) - raw logits
        >>> target = torch.ones_like(output)  # (B, 1, H, W) with values 0 or 1
        >>> loss = criterion(output, target)
        >>> loss.backward()
    """
    def __init__(self, smooth=1.0, sigmoid=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid

    def forward(self, inputs, targets):
        """
        Calculate the Dice Loss between predicted logits and targets.

        Args:
            inputs (torch.Tensor): The prediction logits (unbounded).
            targets (torch.Tensor): The target values (0 or 1).

        Returns:
            torch.Tensor: The computed Dice Loss.
        """
        # Apply sigmoid to logits if specified
        if self.sigmoid:
            inputs = torch.sigmoid(inputs)

        # Flatten the inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and sums
        intersection = (inputs * targets).sum()
        pred_sum = inputs.sum()
        target_sum = targets.sum()

        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) /\
            (pred_sum + target_sum + self.smooth)

        # Return Dice Loss
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined BCE and Dice Loss for binary segmentation tasks.

    This class combines Binary Cross Entropy and Dice Loss with a specified
        weight.
    The combination often leads to better convergence and results compared to
    using either loss on its own.

    Attributes:
        bce_weight (float): Weight for the BCE Loss component (0-1)
        dice_weight (float): Weight for the Dice Loss component (0-1)

    Example:
        >>> criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        >>> output = model(input)  # shape: (B, 1, H, W) - raw logits
        >>> target = torch.ones_like(output)  # (B, 1, H, W) with values 0 or 1
        >>> loss = criterion(output, target)
        >>> loss.backward()
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

    def forward(self, inputs, targets):
        """
        Calculate the Combined BCE and Dice Loss between predicted logits and
            targets.

        Args:
            inputs (torch.Tensor): The prediction logits (unbounded).
            targets (torch.Tensor): The target values (0 or 1).

        Returns:
            torch.Tensor: The weighted sum of BCE Loss and Dice Loss.
        """
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice
