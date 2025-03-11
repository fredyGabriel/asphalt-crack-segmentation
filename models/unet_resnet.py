import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List


class ResNetEncoder(nn.Module):
    """
    ResNet-based encoder with extraction of intermediate features
    for skip connections.
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()

        # Mapping of available models
        resnet_models = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }

        if backbone not in resnet_models:
            raise ValueError(f"Backbone {backbone} not supported. Options: \
{list(resnet_models.keys())}")

        # Load pre-trained model
        self.backbone_name = backbone
        encoder = resnet_models[backbone](weights='IMAGENET1K_V1' if
                                          pretrained else None)

        # Register output channels for each layer
        self._out_channels = {
            "resnet18": (64, 64, 128, 256, 512),
            "resnet34": (64, 64, 128, 256, 512),
            "resnet50": (64, 256, 512, 1024, 2048),
            "resnet101": (64, 256, 512, 1024, 2048),
            "resnet152": (64, 256, 512, 1024, 2048),
        }[backbone]

        # Extract ResNet layers for encoder use
        self.layer0 = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
        )
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1  # 64 or 256 channels
        self.layer2 = encoder.layer2  # 128 or 512 channels
        self.layer3 = encoder.layer3  # 256 or 1024 channels
        self.layer4 = encoder.layer4  # 512 or 2048 channels

        # Register layers for gradual unfreezing
        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3,
                       self.layer4]

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass that returns intermediate features for skip connections.
        """
        features = []

        # Stem block and first feature
        x0 = self.layer0(x)
        features.append(x0)

        # First max pooling
        x = self.maxpool(x0)

        # Capture outputs from each residual block
        x1 = self.layer1(x)
        features.append(x1)

        x2 = self.layer2(x1)
        features.append(x2)

        x3 = self.layer3(x2)
        features.append(x3)

        x4 = self.layer4(x3)
        features.append(x4)

        return features

    def freeze(self):
        """Freezes all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_gradually(self, stage: int = 0):
        """
        Gradually unfreezes the encoder in stages.
        Stage 0: Everything frozen
        Stage 1: layer4 unfrozen
        Stage 2: layer3 and layer4 unfrozen
        Stage 3: layer2, layer3 and layer4 unfrozen
        Stage 4: layer1, layer2, layer3 and layer4 unfrozen
        Stage 5: Everything unfrozen
        """
        # First freeze everything
        self.freeze()

        # Then unfreeze according to stage
        stages = [
            [],  # Stage 0: everything frozen
            [self.layer4],  # Stage 1
            [self.layer3, self.layer4],  # Stage 2
            [self.layer2, self.layer3, self.layer4],  # Stage 3
            [self.layer1, self.layer2, self.layer3, self.layer4],  # Stage 4
            [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]
        ]

        if 0 <= stage < len(stages):
            for layer in stages[stage]:
                for param in layer.parameters():
                    param.requires_grad = True

            print(f"Unfrozen {len(stages[stage])} encoder layers \
(Stage {stage})")
        else:
            raise ValueError(f"Invalid stage {stage}. Must be between 0 and \
{len(stages)-1}.")


class DecoderBlock(nn.Module):
    """
    Basic decoder block with upsampling, skip connection and dropout.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 dropout_p: float = 0.0):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )

        # Convolutions after concatenation with dropout
        conv_layers = [
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        # Add dropout after the first block if specified
        if dropout_p > 0:
            conv_layers.append(nn.Dropout2d(p=dropout_p))

        conv_layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])

        # Add dropout after the second block if specified
        if dropout_p > 0:
            conv_layers.append(nn.Dropout2d(p=dropout_p))

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsampling
        x = self.upsample(x)

        # Adjust dimensions if necessary (padding center-crop)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                              align_corners=False)

        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)

        # Apply convolutions with dropout
        return self.conv(x)


class UNetResNet(nn.Module):
    """
    U-Net with ResNet encoder, gradual unfreezing, and configurable dropout.
    """
    def __init__(
        self,
        backbone: str = "resnet50",
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained: bool = True,
        freeze_encoder: bool = True,
        features: List[int] = [256, 128, 64, 32],
        decoder_dropout: float = 0.1,  # Base dropout rate for decoder
        dropout_factor: float = 1.0,  # Factor to scale dropout by level
        image_size: int = 256  # Target image size for upsampling
    ):
        super().__init__()

        self.encoder = ResNetEncoder(backbone, pretrained)
        encoder_channels = self.encoder.out_channels

        # Freeze encoder if specified
        if freeze_encoder:
            self.encoder.freeze()

        # Define dropout rates for each decoder level
        # Deeper = more dropout, similar to unet2.py
        dropout_rates = [
            decoder_dropout * dropout_factor * (4-i) for i in range(4)
        ]

        # Create decoder blocks with decreasing dropout rates
        self.decoder1 = DecoderBlock(encoder_channels[4], encoder_channels[3],
                                     features[0], dropout_p=dropout_rates[0])
        self.decoder2 = DecoderBlock(features[0], encoder_channels[2],
                                     features[1], dropout_p=dropout_rates[1])
        self.decoder3 = DecoderBlock(features[1], encoder_channels[1],
                                     features[2], dropout_p=dropout_rates[2])
        self.decoder4 = DecoderBlock(features[2], encoder_channels[0],
                                     features[3], dropout_p=dropout_rates[3])

        # Final layer (no dropout before output)
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[3], out_channels, kernel_size=1),
        )

        # Upsample to ensure correct output size
        self.upsample = nn.Upsample(size=(image_size, image_size),
                                    mode='bilinear', align_corners=True)

        # Unfreezing state
        self.current_unfreeze_stage = 0 if freeze_encoder else 5
        self.best_iou = 0.0
        self.consecutive_no_improve = 0

        # Save dropout configuration for reference
        self.dropout_config = {
            'base_rate': decoder_dropout,
            'factor': dropout_factor,
            'rates_by_level': dropout_rates
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoding - extracting features from different levels
        features = self.encoder(x)

        # Decoding with skip connections and dropout
        x = self.decoder1(features[4], features[3])
        x = self.decoder2(x, features[2])
        x = self.decoder3(x, features[1])
        x = self.decoder4(x, features[0])

        # Final prediction
        x = self.final_conv(x)
        x = self.upsample(x)  # Ensure the output has the correct size
        return x

    def unfreeze_next_stage_if_needed(self, current_iou: float,
                                      patience: int = 3):
        """
        Unfreezes the next encoder stage if the IoU metric
        hasn't improved for 'patience' consecutive validations.
        """
        # If IoU improves, update best value and reset counter
        if current_iou > self.best_iou:
            self.best_iou = current_iou
            self.consecutive_no_improve = 0
            return False  # Nothing was unfrozen

        # If IoU doesn't improve, increment counter
        self.consecutive_no_improve += 1

        # If we've gone 'patience' validations without improvement,
        # unfreeze next stage
        if self.consecutive_no_improve >= patience:
            # If everything is already unfrozen, do nothing
            if self.current_unfreeze_stage >= 5:
                return False

            # Unfreeze next stage
            self.current_unfreeze_stage += 1
            self.encoder.unfreeze_gradually(self.current_unfreeze_stage)
            self.consecutive_no_improve = 0  # Reset counter
            return True  # Indicate that a stage was unfrozen

        return False  # Nothing was unfrozen


if __name__ == "__main__":
    # Test the model with dropout
    model = UNetResNet(backbone="resnet50", pretrained=True,
                       freeze_encoder=True, decoder_dropout=0.2)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Encoder frozen: {not any(p.requires_grad for p in
                                     model.encoder.parameters())}")
    print(f"Dropout config: {model.dropout_config}")

    # Test unfreezing
    model.unfreeze_next_stage_if_needed(current_iou=0.5, patience=1)
    print(f"Current stage: {model.current_unfreeze_stage}")

    # Simulate several epochs without improvement to test unfreezing
    model.unfreeze_next_stage_if_needed(current_iou=0.5, patience=1)
    print(f"Current stage: {model.current_unfreeze_stage}")
    print(f"Layer4 requires gradient: \
{any(p.requires_grad for p in model.encoder.layer4.parameters())}")
