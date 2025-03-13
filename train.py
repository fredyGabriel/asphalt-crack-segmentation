import torch
import numpy as np
from utils.utils import load_config
from data.data_loader import get_data_loaders
from factory.model_manager import ModelManager
from factory.trainer import Trainer
from factory.losses import BinaryCrossEntropyLoss, DiceLoss, CombinedLoss


def main():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Centralized configuration loading
    config = load_config('configs/default.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # Create model manager
    model_manager = ModelManager(config, device)
    model_manager.create_model()

    # Loss function - using config directly here
    loss_config = config['model'].get('loss', {})
    loss_type = loss_config.get('type', 'bce').lower()

    if loss_type == 'dice':
        criterion = DiceLoss(sigmoid=True)
        print("Using Dice Loss")
    elif loss_type == 'combined':
        bce_weight = loss_config.get('bce_weight', 0.5)
        dice_weight = loss_config.get('dice_weight', 0.5)
        criterion = CombinedLoss(bce_weight=bce_weight,
                                 dice_weight=dice_weight)
        print(f"Using Combined Loss (BCE weight: {bce_weight}, \
Dice weight: {dice_weight})")
    else:  # default to bce
        criterion = BinaryCrossEntropyLoss()
        print("Using Binary Cross Entropy Loss")

    # Set up optimizer using model_manager
    base_lr = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 1e-4)
    optimizer = model_manager.setup_optimizer(base_lr, weight_decay)

    # Set up scheduler using model_manager
    scheduler = model_manager.setup_scheduler(optimizer)

    # Create trainer and pass the config
    trainer = Trainer(
        model_manager=model_manager,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,  # Pass the same config object to all components
        device=device
    )

    # Run training loop
    trainer.train()

    # Evaluate on test set - already using the same config from trainer
    trainer.evaluate(test_loader)


if __name__ == '__main__':
    main()
