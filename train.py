import os
import yaml
import torch
import numpy as np

from torch.amp import GradScaler
from data.dataset import CrackDataset, TransformedSubset
from data.transforms import (Compose, ToTensor, Resize, Normalize,
                             RandomNoise, RandomShadow)
# Import both model architectures
from models.unet import UNet
from models.unet2 import UNet as UNet2
from models.unet_resnet import UNetResNet

from factory.losses import BinaryCrossEntropyLoss, DiceLoss, CombinedLoss
from validators.evaluate import validate, calculate_metrics
from factory.train_one import train_one_epoch


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_model(config, device):
    """
    Create a model based on the configuration.

    Args:
        config: Configuration dictionary with model parameters
        device: Device to move the model to

    Returns:
        Configured model instance
    """
    model_type = config['model'].get('type', 'unet').lower()
    input_channels = config['model'].get('input_channels', 3)
    output_channels = config['model'].get('output_channels', 1)

    if model_type == "unet":
        # Check if we should use UNet or UNet2 (with dropout)
        unet_config = config['model'].get('unet', {})

        if 'encoder_dropout' in unet_config:
            # Use UNet2 with dropout parameters
            model = UNet2(
                in_channels=input_channels,
                out_channels=output_channels,
                encoder_dropout=unet_config.get('encoder_dropout', 0.1),
                bottleneck_dropout=unet_config.get('bottleneck_dropout', 0.5),
                decoder_dropout=unet_config.get('decoder_dropout', 0.1)
            )
            print("Created UNet2 model with dropout")
        else:
            # Use standard UNet
            model = UNet(
                in_channels=input_channels,
                out_channels=output_channels
            )
            print("Created standard UNet model")

    elif model_type == "unet_resnet":
        # Use UNetResNet with ResNet backbone
        resnet_config = config['model'].get('unet_resnet', {})

        model = UNetResNet(
            backbone=resnet_config.get('backbone', 'resnet50'),
            in_channels=input_channels,
            out_channels=output_channels,
            pretrained=resnet_config.get('pretrained', True),
            freeze_encoder=resnet_config.get('freeze_encoder', True),
            features=resnet_config.get('features', [256, 128, 64, 32]),
            decoder_dropout=resnet_config.get('decoder_dropout', 0.1),
            dropout_factor=resnet_config.get('dropout_factor', 1.0)
        )
        print(f"Created UNetResNet with \
{resnet_config.get('backbone', 'resnet50')} backbone")
    else:
        raise ValueError(f"Unknown model type: {model_type}. \
Supported types: 'unet', 'unet_resnet'")

    return model.to(device)


def main():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    config = load_config('configs/default.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define how often to save metrics and model checkpoints (every N epochs)
    save_frequency = config.get('training', {}).get('save_frequency', 5)
    print(f"Saving metrics and checkpoints every {save_frequency} epochs")

    # Cargar tamaño de imagen desde la configuración
    image_size = config.get('training', {}).get('image_size', 256)

    dataset_path = config['paths']['dataset_path']

    base_dataset = CrackDataset(
        images_dir=dataset_path + '/images',
        masks_dir=dataset_path + '/masks',
        transform=None)  # No transform for now

    generator1 = torch.Generator().manual_seed(42)
    base_train, base_val, base_test = torch.utils.data.random_split(
        base_dataset, [.70, .15, .15], generator1)

    # Definir transformaciones
    train_transforms = Compose([
        Resize((image_size, image_size)),  # First resize
        ToTensor(),  # Second convert to tensor
        RandomNoise(),
        RandomShadow(),
        Normalize(),  # Finally normalize
    ])

    val_transforms = Compose([
        Resize((image_size, image_size)),  # First resize
        ToTensor(),  # Second convert to tensor
        Normalize(),  # Finally normalize
    ])

    train = TransformedSubset(base_train, train_transforms)
    val = TransformedSubset(base_val, val_transforms)
    test = TransformedSubset(base_test, val_transforms)

    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    epochs = config['training']['num_epochs']
    num_workers = config['training']['num_workers']

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Speeds up host to GPU transfers
        drop_last=True,   # Prevents issues with small last batches
        persistent_workers=True  # Keeps workers alive between epochs
    )
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    # Create model based on configuration
    model = create_model(config, device)

    # Check if we're using UNetResNet for unfreezing functionality
    model_type = config['model'].get('type', 'unet').lower()
    is_unet_resnet = model_type == "unet_resnet"

    # Get unfreezing patience if using UNetResNet
    unfreezing_patience = 5
    if is_unet_resnet:
        unfreezing_patience = config['model'].get('unet_resnet', {}).\
            get('unfreezing_patience', 5)

    # Función de pérdida
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
        print(f"Using Combined Loss (BCE weight: {bce_weight}, Dice weight: \
{dice_weight})")
    else:  # default to bce
        criterion = BinaryCrossEntropyLoss()
        print("Using Binary Cross Entropy Loss")

    weight_decay = config['training'].get('weight_decay', 1e-4)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    # Crear scheduler dinámicamente basado en la configuración
    scheduler_config = config.get('training', {}).get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau')

    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 5),
            min_lr=scheduler_config.get('min_lr', 0.00001)
        )
    # Añadir otros tipos de schedulers según sea necesario

    # En train.py
    use_mixed_precision = config.get('training', {}).get('mixed_precision',
                                                         True)
    scaler = GradScaler() if use_mixed_precision and device.type == 'cuda'\
        else None

    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'iou': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'epochs': []  # Store which epochs were recorded
    }

    # Tracking best model
    best_val_loss = float('inf')
    best_iou = 0.0

    os.makedirs(config['model']['save_path'], exist_ok=True)

    # Early stopping configuration
    early_stopping_patience = config.get('training',
                                         {}).get('early_stopping_patience', 10
                                                 )
    early_stopping_counter = 0

    # Show execution time per epoch
    import time
    start_time = time.time()

    # Use TensorBoard for visualization
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=config['model']['save_path'])

    # Después de cargar config
    accumulation_steps = config.get('training', {}).get('accumulation_steps',
                                                        4)

    for epoch in range(epochs):
        # Add torch.cuda.empty_cache() at the beginning of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # En el bucle de entrenamiento
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, accumulation_steps=accumulation_steps
        )
        val_loss = validate(model, val_loader, criterion, device)

        # Calculate metrics on validation set AFTER epoch
        metrics = calculate_metrics(model, val_loader, device, config)

        # Check if we should unfreeze the next stage for UNetResNet
        if is_unet_resnet and hasattr(model, 'unfreeze_next_stage_if_needed'):
            did_unfreeze = model.unfreeze_next_stage_if_needed(
                current_iou=metrics['iou'],
                patience=unfreezing_patience
            )
            if did_unfreeze:
                print(f"Unfroze next stage of ResNet encoder (stage \
{model.current_unfreeze_stage})")

        # Determine if we should save metrics in this epoch
        save_epoch = (epoch % save_frequency == 0) or (epoch == epochs - 1)

        # Always show results each epoch to track progress
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, \
Val Loss: {val_loss:.4f}, ' f'iou: {metrics["iou"]:.4f}, f1: \
{metrics["f1"]:.4f}' f'{" - Saving metrics" if save_epoch else ""}')

        # Save metrics only in designated epochs
        if save_epoch:
            history['epochs'].append(epoch + 1)  # Save epoch number (1-idxed)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            for key, value in metrics.items():
                if key in history:
                    history[key].append(value)

            # Also save model checkpoint at these intervals
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if
                scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'metrics': metrics,
                'model_type': model_type  # Save model type for loading
            }, os.path.join(config['model']['save_path'],
                            f'checkpoint_epoch_{epoch+1}.pth'))

        # Save best model by loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0

            # Save best model here (within the same condition)
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': model_type,
                'val_loss': val_loss,
                'epoch': epoch + 1
            }, os.path.join(config['model']['save_path'],
                            'best_loss_model.pth'))
            print(f"Saved new best model by loss (epoch {epoch+1})")
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/\
{early_stopping_patience}")
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Save best model by IoU
        if metrics['iou'] > best_iou:
            best_iou = metrics['iou']

            # Save model with state and metrics
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'metrics': metrics,
                'model_type': model_type  # Save model type for loading
            }, os.path.join(config['model']['save_path'],
                            'best_iou_model.pth'))

            print(f"Saved new best model by IoU (epoch {epoch+1}, IoU: \
{metrics['iou']:.4f})")

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Calculate epoch time
        epoch_time = time.time() - start_time
        print(f"Epoch time: {epoch_time:.2f}s")
        start_time = time.time()

        # In training loop:
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/iou', metrics['iou'], epoch)
        writer.add_scalar('Metrics/f1', metrics['f1'], epoch)
        writer.add_scalar('Metrics/precision', metrics['precision'], epoch)
        writer.add_scalar('Metrics/recall', metrics['recall'], epoch)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'epoch': epochs
    }, os.path.join(config['model']['save_path'], 'last_model.pth'))

    # Save metrics history
    import json
    with open(os.path.join(config['model']['save_path'],
                           'training_history.json'), 'w') as f:
        # Convert values to list before saving
        for key in history:
            history[key] = [float(val) if not isinstance(val, int) else val
                            for val in history[key]]
        json.dump(history, f, indent=4)

    print("Training completed. History and models saved.")

    # After training:
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True  # Same as train
    )

    # Load the best model by iou
    best_model_path = os.path.join(config['model']['save_path'],
                                   'best_iou_model.pth')
    checkpoint = torch.load(best_model_path)

    # Create a new model of the same type for testing
    model = create_model(config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Calculate final metrics on test set
    test_metrics = calculate_metrics(model, test_loader, device, config)

    # Calculate loss on test set
    test_loss = validate(model, test_loader, criterion, device)

    print("\nFinal evaluation on test set:")
    print(f"Test Loss: {test_loss:.4f}")
    for metric, value in test_metrics.items():
        print(f"Test {metric}: {value:.4f}")

    # Save test results
    with open(os.path.join(config['model']['save_path'],
                           'test_results.json'), 'w') as f:
        results = {
            "test_loss": float(test_loss),
            "model_type": model_type
        }
        results.update({k: float(v) for k, v in test_metrics.items()})
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
