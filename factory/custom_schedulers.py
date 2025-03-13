import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Custom scheduler that combines linear warmup with cosine annealing.
    Supports different learning rates for encoder and decoder parameter groups.
    """
    def __init__(self, optimizer, T_warmup, T_max, eta_min_decoder=0,
                 eta_min_encoder=0, last_epoch=-1, verbose=False):
        """
        Args:
            optimizer: Optimizer with parameter groups
            T_warmup: Number of warmup epochs
            T_max: Total number of epochs (including warmup)
            eta_min_decoder: Minimum learning rate for decoder after annealing
            eta_min_encoder: Minimum learning rate for encoder after annealing
            last_epoch: Last epoch (-1 means start fresh)
            verbose: Whether to print LR updates
        """
        self.T_warmup = T_warmup
        self.T_max = T_max
        self.eta_min_decoder = eta_min_decoder
        self.eta_min_encoder = eta_min_encoder
        self.base_lrs_by_group = []  # Store target LR for each group
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch,
                                                      verbose)

    def get_lr(self):
        # Different behavior depending on whether we're in warmup or annealing
        # phase
        if self.last_epoch < self.T_warmup:
            # Warmup phase: linear increase
            return [self._get_warmup_lr(base_lr, group_idx) for group_idx,
                    base_lr in enumerate(self.base_lrs)]
        else:
            # Cosine annealing phase
            return [self._get_cosine_lr(base_lr, group_idx) for group_idx,
                    base_lr in enumerate(self.base_lrs)]

    def _get_warmup_lr(self, base_lr, group_idx):
        # Linear warmup from min_lr to target_lr
        min_lr = self.eta_min_encoder if group_idx == 0 else\
            self.eta_min_decoder
        return min_lr + (base_lr - min_lr) * (self.last_epoch / self.T_warmup)

    def _get_cosine_lr(self, base_lr, group_idx):
        # Standard cosine annealing, but starting after warmup
        min_lr = self.eta_min_encoder if group_idx == 0 else\
            self.eta_min_decoder
        progress = (self.last_epoch - self.T_warmup) /\
            (self.T_max - self.T_warmup)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_factor


def setup_scheduler(self, optimizer):
    """
    Set up learning rate scheduler based on configuration.

    Args:
        optimizer: The optimizer to schedule

    Returns:
        torch.optim.lr_scheduler: Configured scheduler
    """
    scheduler_config = self.config.get('training', {}).get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau')

    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 5),
            min_lr=scheduler_config.get('min_lr', 0.00001)
        )
        print(f"Created ReduceLROnPlateau scheduler with patience=\
{scheduler_config.get('patience', 5)}")

    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get('T_0', 10),
            T_mult=scheduler_config.get('T_mult', 1),
            eta_min=scheduler_config.get('min_lr', 0.00001)
        )
        print(f"Created CosineAnnealingWarmRestarts scheduler with T_0=\
{scheduler_config.get('T_0', 10)}")

    elif scheduler_type == 'WarmupCosineAnnealing':
        # Get configuration parameters with appropriate defaults
        warmup_epochs = scheduler_config.get('warmup_epochs', 5)
        total_epochs = self.config['training'].get('num_epochs', 100)
        min_lr_decoder = scheduler_config.get('min_lr_decoder', 0.00001)
        min_lr_encoder = scheduler_config.get('min_lr_encoder', 0.000001)

        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            T_warmup=warmup_epochs,
            T_max=total_epochs,
            eta_min_decoder=min_lr_decoder,
            eta_min_encoder=min_lr_encoder
        )
        print(f"Created WarmupCosineAnnealing scheduler: warmup=\
{warmup_epochs}, "
              f"total={total_epochs}, min_lr_decoder={min_lr_decoder}, "
              f"min_lr_encoder={min_lr_encoder}")
    else:
        print(f"Warning: Unknown scheduler type '{scheduler_type}', \
using constant learning rate")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lambda epoch: 1.0)

    return scheduler
