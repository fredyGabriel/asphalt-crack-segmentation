import torch

# Import models
from models.unet import UNet
from models.unet2 import UNet2
from models.unet_resnet import UNetResNet
from models.swin2_unet import SwinUNetV2


def get_model(config, logger=None):
    """
    Create a model based on configuration

    Args:
        config: Configuration dictionary
        logger: Optional logger for messages

    Returns:
        nn.Module: Model instance
    """
    def log(message):
        """Helper function to log messages"""
        if logger:
            logger.log_info(message)
        else:
            print(message)

    if "model" not in config:
        raise ValueError("Missing 'model' section in configuration")

    model_type = config["model"].get("type")
    if not model_type:
        raise ValueError("Model type not specified in configuration")

    # Get standardized parameters from config
    in_channels = config["model"].get("in_channels", 3)
    out_channels = config["model"].get("out_channels", 1)

    if model_type == "unet":
        # Create UNet model
        unet_config = config["model"].get("unet", {})
        model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters=unet_config.get("num_filters", 64),
            encoder_dropout=unet_config.get("encoder_dropout", 0.1),
            bottleneck_dropout=unet_config.get("bottleneck_dropout", 0.5),
            decoder_dropout=unet_config.get("decoder_dropout", 0.1)
        )
    elif model_type == "unet2":
        # Added support for UNet2
        unet2_config = config["model"].get("unet2", {})
        model = UNet2(
            in_channels=in_channels,
            out_channels=out_channels,
            features=unet2_config.get("features", [64, 128, 256, 512]),
            dropout=unet2_config.get("dropout", 0.1),
            use_attention=unet2_config.get("use_attention", False)
        )
    elif model_type == "unet_resnet":
        # Proper implementation of UNetResNet creation
        resnet_config = config["model"].get("unet_resnet", {})
        model = UNetResNet(
            backbone=resnet_config.get("backbone", "resnet50"),
            in_channels=in_channels,
            out_channels=out_channels,
            pretrained=resnet_config.get("pretrained", True),
            freeze_encoder=resnet_config.get("freeze_encoder", True),
            features=resnet_config.get("features", [256, 128, 64, 32]),
            decoder_dropout=resnet_config.get("decoder_dropout", 0.1),
            dropout_factor=resnet_config.get("dropout_factor", 1.0),
            image_size=resnet_config.get("image_size", 256)
        )
    elif model_type == "swin2_unet":
        # Create SwinUNetV2 model with standardized parameter names
        swin_config = config['model'].get('swin2_unet', {})

        model = SwinUNetV2(
            img_size=swin_config.get("img_size", 224),
            patch_size=swin_config.get("patch_size", 4),
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=swin_config.get("embed_dim", 96),
            depths=swin_config.get("depths", [2, 2, 6, 2]),
            depths_decoder=swin_config.get("depths_decoder", [2, 2, 2, 2]),
            num_heads=swin_config.get("num_heads", [3, 6, 12, 24]),
            window_size=swin_config.get("window_size", 7),
            pretrained=swin_config.get("pretrained", False),
            pretrained_model_path=swin_config.get("pretrained_model_path",
                                                  None),
            freeze_encoder=swin_config.get("freeze_encoder", True),
            decoder_dropout=swin_config.get("decoder_dropout", 0.0),
            aspp_rates=swin_config.get("aspp_rates", [3, 6, 9])
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    log(f"Created {model_type} model with {in_channels} input channels and \
{out_channels} output channels")
    return model


def load_model_from_checkpoint(checkpoint_path, config=None, logger=None):
    """
    Load model from checkpoint

    Args:
        checkpoint_path: Path to the checkpoint file
        config: Directly provided configuration (prioritized)
        logger: Optional logger for messages

    Returns:
        nn.Module: Model with loaded weights
    """
    def log(message):
        """Helper function to log messages"""
        if logger:
            logger.log_info(message)
        else:
            print(message)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    log(f"Loading model from checkpoint: {checkpoint_path}")

    # Get config (prioritizing directly provided config)
    if config is None:
        if 'config' in checkpoint:
            config = checkpoint['config']
            log("Using configuration from checkpoint")
        else:
            raise ValueError("No configuration found in checkpoint and no \
config provided")

    # Create model
    model = get_model(config, logger)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    log("Model weights loaded successfully")
    return model


if __name__ == "__main__":
    # Example usage
    from utils.utils import load_config
    from utils.logger import Logger

    config_path = "configs/default.yaml"
    config = load_config(config_path)

    # Initialize logger
    logger = Logger(log_dir="logs")

    # Create a model based on config
    model = get_model(config, logger)

    # Example of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.log_info(f"Total parameters: {total_params:,}")

    trainable_params = sum(p.numel() for p in model.parameters() if
                           p.requires_grad)
    logger.log_info(f"Trainable parameters: {trainable_params:,}")

    # Close logger
    logger.close()
