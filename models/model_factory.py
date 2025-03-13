import torch

# Import models
from models.unet import UNet
from models.unet2 import UNet2  # Added import for UNet2
from models.unet_resnet import UNetResNet
from models.swin2_unet import SwinUNetV2


def get_model(config):
    """
    Create a model based on configuration

    Args:
        config: Configuration dictionary

    Returns:
        nn.Module: Model instance
    """
    if "model" not in config:
        raise ValueError("Missing 'model' section in configuration")

    model_type = config["model"].get("type")
    if not model_type:
        raise ValueError("Model type not specified in configuration")

    # Standardized parameter extraction
    in_channels = config["model"].get("input_channels", 3)  # Standardized name
    out_channels = config["model"].get("output_channels", 1)  # Standar name

    if model_type == "unet":
        # Create UNet model
        unet_config = config["model"].get("unet", {})
        return UNet(
            input_channels=in_channels,
            output_channels=out_channels,
            num_filters=unet_config.get("num_filters", 64),
            encoder_dropout=unet_config.get("encoder_dropout", 0.1),
            bottleneck_dropout=unet_config.get("bottleneck_dropout", 0.5),
            decoder_dropout=unet_config.get("decoder_dropout", 0.1)
        )

    elif model_type == "unet2":
        # Added support for UNet2
        unet2_config = config["model"].get("unet2", {})
        return UNet2(
            input_channels=in_channels,  # Using standardized parameter name
            output_channels=out_channels,  # Using standardized parameter name
            features=unet2_config.get("features", [64, 128, 256, 512]),
            dropout=unet2_config.get("dropout", 0.1),
            use_attention=unet2_config.get("use_attention", False)
        )

    elif model_type == "unet_resnet":
        # Proper implementation of UNetResNet creation
        resnet_config = config["model"].get("unet_resnet", {})

        # Direct instantiation instead of relying on static method
        return UNetResNet(
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

        return SwinUNetV2(
            img_size=swin_config.get("img_size", 224),
            patch_size=swin_config.get("patch_size", 4),
            in_chans=in_channels,  # Using standardized parameter
            num_classes=out_channels,  # Using standardized parameter
            embed_dim=swin_config.get("embed_dim", 96),
            depths=swin_config.get("depths", [2, 2, 6, 2]),
            depths_decoder=swin_config.get("depths_decoder", [2, 2, 2, 2]),
            num_heads=swin_config.get("num_heads", [3, 6, 12, 24]),
            window_size=swin_config.get("window_size", 7),
            pretrained=swin_config.get("pretrained", False),
            pretrained_model_path=swin_config.get("pretrained_model_path",
                                                  None),
            freeze_encoder=swin_config.get("freeze_encoder", True),
            decoder_dropout=swin_config.get("decoder_dropout", 0.0)
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model_from_checkpoint(checkpoint_path, config=None):
    """
    Load model from checkpoint

    Args:
        checkpoint_path: Path to the checkpoint file
        config: Directly provided configuration (prioritized)

    Returns:
        nn.Module: Model with loaded weights
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get config (prioritizing directly provided config)
    if config is None:
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("No configuration found in checkpoint and no \
config provided")

    # Create model
    model = get_model(config)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    return model


if __name__ == "__main__":
    # Example usage
    from utils.utils import load_config
    config_path = "configs/default.yaml"
    config = load_config(config_path)

    # Create a model based on config
    model = get_model(config)
    print(f"Created model: {type(model).__name__}")

    # Example of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    trainable_params = sum(p.numel() for p in model.parameters() if
                           p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
