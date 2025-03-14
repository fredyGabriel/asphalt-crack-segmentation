import torch
from utils.utils import load_config
from data.data_loader import get_data_loaders
from factory.model_manager import ModelManager
from factory.trainer import Trainer
from factory.losses import BinaryCrossEntropyLoss, DiceLoss, CombinedLoss
from utils.logger import Logger


def main():
    # Centralized configuration loading
    config = load_config('configs/default.yaml')

    # Initialize logger
    log_dir = config['model'].get('save_path', 'saved_models/default')
    logger = Logger(log_dir=log_dir, config=config)
    logger.log_info("Starting training process")

    # Set seeds for reproducibility from config
    seed = config['training'].get('seed', 42)  # Default to 42 if not specified
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    logger.log_info(f"Using random seed: {seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log_info(f"Using device: {device}")

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # Create model manager
    model_manager = ModelManager(config, device, logger)
    model_manager.create_model()

    # Accelerate training on RTX GPUs
    model_manager.model = torch.compile(model_manager.model)

    # Adjust cudnn to improve performance
    torch.backends.cudnn.benchmark = True

    # Loss function - using config directly here
    loss_config = config['model'].get('loss', {})
    loss_type = loss_config.get('type', 'bce').lower()

    if loss_type == 'dice':
        criterion = DiceLoss(sigmoid=True)
        logger.log_info("Using Dice Loss")
    elif loss_type == 'combined':
        bce_weight = loss_config.get('bce_weight', 0.5)
        dice_weight = loss_config.get('dice_weight', 0.5)
        criterion = CombinedLoss(bce_weight=bce_weight,
                                 dice_weight=dice_weight)
        logger.log_info(f"Using Combined Loss (BCE weight: {bce_weight}, "
                        f"Dice weight: {dice_weight})")
    else:  # default to bce
        criterion = BinaryCrossEntropyLoss()
        logger.log_info("Using Binary Cross Entropy Loss")

    # Set up optimizer using model_manager
    base_lr = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 1e-4)
    optimizer = model_manager.setup_optimizer(base_lr, weight_decay)

    # Set up scheduler using model_manager
    scheduler = model_manager.setup_scheduler(optimizer)

    # Create trainer and pass the config and logger
    trainer = Trainer(
        model_manager=model_manager,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        logger=logger  # Pass the logger to trainer
    )

    # Run training loop
    trainer.train()

    # Evaluate on test set - already using the same config from trainer
    trainer.evaluate(test_loader)

    # Close logger
    logger.close()


if __name__ == '__main__':
    main()
