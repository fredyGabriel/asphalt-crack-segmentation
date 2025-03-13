import torch
from torch.utils.data import DataLoader
from data.dataset import CrackDataset, TransformedSubset
from data.transforms import (Compose, ToTensor, Resize, Normalize,
                             RandomNoise, RandomShadow)


def get_dataset_splits(config, seed=42):
    """
    Create dataset splits (train, validation, test) based on configuration.

    Args:
        config (dict): Configuration dictionary
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (base_train, base_val, base_test) - Dataset splits
    """
    dataset_path = config['paths']['dataset_path']

    # Create the base dataset
    base_dataset = CrackDataset(
        images_dir=dataset_path + '/images',
        masks_dir=dataset_path + '/masks',
        transform=None  # No transform for now
    )

    # Split the dataset
    generator = torch.Generator().manual_seed(seed)
    base_train, base_val, base_test = torch.utils.data.random_split(
        base_dataset, [.70, .15, .15], generator
    )

    return base_train, base_val, base_test


def get_transforms(image_size):
    """
    Create data transformations for training and validation/testing.

    Args:
        image_size (int): Target image size

    Returns:
        tuple: (train_transforms, val_transforms) - Transformation pipelines
    """
    # Define transformations for training (with augmentation)
    train_transforms = Compose([
        Resize((image_size, image_size)),  # First resize
        ToTensor(),                        # Convert to tensor
        RandomNoise(),                     # Add random noise
        RandomShadow(),                    # Add random shadow
        Normalize(),                       # Normalize
    ])

    # Define transformations for validation/testing (no augmentation)
    val_transforms = Compose([
        Resize((image_size, image_size)),  # First resize
        ToTensor(),                        # Convert to tensor
        Normalize(),                       # Normalize
    ])

    return train_transforms, val_transforms


def get_transformed_datasets(base_train, base_val, base_test, image_size):
    """
    Apply transformations to dataset splits.

    Args:
        base_train, base_val, base_test: Dataset splits
        image_size (int): Target image size

    Returns:
        tuple: (train, val, test) - Transformed datasets
    """
    # Get transforms
    train_transforms, val_transforms = get_transforms(image_size)

    # Apply transforms
    train = TransformedSubset(base_train, train_transforms)
    val = TransformedSubset(base_val, val_transforms)
    test = TransformedSubset(base_test, val_transforms)

    return train, val, test


def get_data_loaders(config):
    """
    Create data loaders for training, validation, and testing.

    Args:
        config (dict): Configuration dictionary

    Returns:
        tuple: (train_loader, val_loader, test_loader) - Data loaders
    """
    # Get dataset splits
    base_train, base_val, base_test = get_dataset_splits(config)

    # Get image size from configuration
    image_size = config.get('training', {}).get('image_size', 256)

    # Transform datasets
    train, val, test = get_transformed_datasets(
        base_train, base_val, base_test, image_size
    )

    # Get loader parameters from configuration
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']

    # Create data loaders
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,          # Speeds up host to GPU transfers
        drop_last=True,           # Prevents issues with small last batches
        persistent_workers=True    # Keeps workers alive between epochs
    )

    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
