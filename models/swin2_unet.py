import os
import torch
from collections import OrderedDict
import torch.nn as nn
from models.swin_transformer_v2 import SwinTransformerV2


class PatchExpand(nn.Module):
    """
    Patch Expanding Layer: inverse of PatchMerging, expands spatial resolution
    """
    def __init__(self, input_resolution, dim, dim_scale=2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, dim_scale*dim, bias=False)
        self.norm = norm_layer(dim_scale*dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.expand(x)
        x = self.norm(x)

        # Reorganize to expand spatial resolution
        x = x.view(B, H, W, 2*C)
        x = x.reshape(B, H, W, 2, C)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(B, H*2, W, C)

        x = x.reshape(B, H*2, W, C)
        x = x.permute(0, 3, 1, 2)  # B C H W
        x = x.view(B, C, H*2, W)

        # Similarly for W dimension
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = x.reshape(B, H*2, W, C)
        x = x.reshape(B, H*2, W, 1, C)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(B, H*2, W*2, C//2)

        x = x.view(B, H*2*W*2, C//2)

        return x


class DecoderBlock(nn.Module):
    """Decoder block that uses multi-scale features from the encoder"""
    def __init__(self, dim_in, dim_out, input_resolution, depth=2, num_heads=3,
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 use_checkpoint=False):
        super().__init__()

        self.patch_expand = PatchExpand(
            input_resolution=input_resolution,
            dim=dim_in,
            dim_scale=2,
            norm_layer=norm_layer
        )

        new_resolution = (input_resolution[0]*2, input_resolution[1]*2)

        # Use encoder layers to process expanded features
        from models.swin_transformer_v2 import BasicLayer
        self.layers = BasicLayer(
            dim=dim_out,
            input_resolution=new_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint
        )

    def forward(self, x, skip=None):
        x = self.patch_expand(x)
        if skip is not None:
            x = torch.cat([x, skip], -1)
        x = self.layers(x)
        return x


class FinalPatchExpand(nn.Module):
    """
    Final Patch Expansion to obtain full image resolution
    """
    def __init__(self, input_resolution, dim, patch_size,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.expand = nn.Linear(dim, patch_size*patch_size*dim, bias=False)
        self.norm = norm_layer(patch_size*patch_size*dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.expand(x)
        x = self.norm(x)

        # Reorganize to expand to full resolution
        x = x.view(B, H, W, C*self.patch_size*self.patch_size)
        x = x.reshape(B, H, W, self.patch_size, self.patch_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, H*self.patch_size, W*self.patch_size, C)

        return x


class SwinUNetV2(nn.Module):
    """
    UNet architecture based on Swin Transformer V2 for crack segmentation
    with transfer learning capabilities.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        depths_decoder=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        pretrained_window_sizes=[0, 0, 0, 0],
        pretrained=False,
        pretrained_model_path=None,
        freeze_encoder=True,
        decoder_dropout=0.0,
        **kwargs
    ):
        super().__init__()

        # Use SwinTransformerV2Encoder as encoder with transfer learning
        # capability
        self.encoder = SwinTransformerV2Encoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            pretrained_window_sizes=pretrained_window_sizes,
            pretrained=pretrained,
            pretrained_model_path=pretrained_model_path
        )

        # Freeze the encoder if specified
        if freeze_encoder:
            self.encoder.freeze()

        # Unfreezing state for fine-tuning
        self.current_unfreeze_stage = 0 if freeze_encoder else len(depths) + 2
        self.best_metric = 0.0
        self.consecutive_no_improve = 0

        # Save important variables for the decoder
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patches_resolution = self.encoder.patches_resolution

        # Create decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        # Build decoder with skip connections
        for i_layer in range(self.num_layers-1, -1, -1):
            # Calculate dropout rate for each level if specified
            dropout_rate = decoder_dropout * (i_layer + 1) / self.num_layers\
                if decoder_dropout > 0 else 0

            layer_up = DecoderBlock(
                dim_in=int(embed_dim * 2 ** i_layer),
                dim_out=int(embed_dim * 2 ** (i_layer-1)) if i_layer > 0 else
                embed_dim,
                input_resolution=(
                    self.patches_resolution[0] // (2 ** i_layer),
                    self.patches_resolution[1] // (2 ** i_layer)
                ),
                depth=depths_decoder[3-i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=dropout_rate,  # Use configurable dropout
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint
            )
            self.layers_up.append(layer_up)

            # Layers to merge features from skip connection
            if i_layer > 0:
                concat_linear = nn.Linear(
                    2*int(embed_dim*2**(i_layer-1)),
                    int(embed_dim*2**(i_layer-1))
                ) if i_layer > 0 else nn.Identity()
                self.concat_back_dim.append(concat_linear)

        # Final layer to expand to full resolution
        self.final_expand = FinalPatchExpand(
            input_resolution=self.patches_resolution,
            dim=embed_dim,
            patch_size=patch_size,
            norm_layer=norm_layer
        )

        # Final segmentation layer
        self.final_conv = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=num_classes,
            kernel_size=1
        )

        # Final activation: sigmoid for binary segmentation
        self.sigmoid = nn.Sigmoid()

    def forward_encoder(self, x):
        """Extract multi-scale features from encoder"""
        return self.encoder(x)

    def forward(self, x):
        # Encode
        bottleneck, encoder_features = self.forward_encoder(x)

        # Decode with skip connections
        x = bottleneck
        for i, layer_up in enumerate(self.layers_up):
            if i == 0:
                x = layer_up(x)
            else:
                skip = encoder_features[i]
                x = layer_up(x, skip)
                x = self.concat_back_dim[i-1](x)

        # Expand to full resolution
        x = self.final_expand(x)

        # Reorganize for final convolution
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # B C H W

        # Final convolution and activation
        x = self.final_conv(x)
        x = self.sigmoid(x)

        return x

    def unfreeze_next_stage_if_needed(self, current_metric, patience=3):
        """
        Unfreezes the next stage of the encoder if the performance metric
        hasn't improved for 'patience' consecutive validations.

        Args:
            current_metric: Current value of the metric (IoU, Dice, etc.)
            patience: Number of validations without improvement before
            unfreezing

        Returns:
            bool: True if a stage was unfrozen, False otherwise
        """
        # If metric improves, update best value and reset counter
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.consecutive_no_improve = 0
            return False  # Nothing was unfrozen

        # If metric doesn't improve, increment counter
        self.consecutive_no_improve += 1

        # If we've gone 'patience' validations without improvement,
        # unfreeze the next stage
        if self.consecutive_no_improve >= patience:
            # Calculate maximum possible stage
            max_stage = len(self.encoder.depths) + 2

            # If everything is already unfrozen, do nothing
            if self.current_unfreeze_stage >= max_stage:
                return False

            # Unfreeze next stage
            self.current_unfreeze_stage += 1
            self.encoder.unfreeze_gradually(self.current_unfreeze_stage)
            self.consecutive_no_improve = 0  # Reset counter
            print(f"Unfrozen stage {self.current_unfreeze_stage} of the \
encoder")
            return True  # Indicate that a stage was unfrozen

        return False  # Nothing was unfrozen


class SwinTransformerV2Encoder(nn.Module):
    """
    Wrapper for the SwinTransformerV2 encoder that adds functionality
    for freezing/unfreezing and loading pre-trained weights.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        pretrained_window_sizes=[0, 0, 0, 0],
        pretrained=False,
        pretrained_model_path=None
    ):
        super().__init__()

        # Create base SwinTransformerV2 model without classification head
        self.model = SwinTransformerV2(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,  # No classification head
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            pretrained_window_sizes=pretrained_window_sizes
        )

        # Save configuration for access in the class
        self.depths = depths
        self.embed_dim = embed_dim
        self.patches_resolution = self.model.patches_resolution

        # Load pre-trained weights if requested
        if pretrained:
            self._load_pretrained_model(pretrained_model_path)

    def _load_pretrained_model(self, pretrained_model_path=None):
        """
        Loads pre-trained weights for the encoder.

        Args:
            pretrained_model_path: Path to the pre-trained model. If None,
                                   tries to use pre-trained weights from
                                   standard sources.
        """
        if pretrained_model_path is None:
            # Here you can define standard paths or use predefined models
            # similar to how torchvision handles pre-trained models
            raise ValueError(
                "Please provide a path to pre-trained weights.")

        if not os.path.isfile(pretrained_model_path):
            raise FileNotFoundError(
                f"File not found: {pretrained_model_path}")

        print(f"Loading pre-trained weights from: {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Filter and adapt weights for the encoder (without classification
        # head)
        encoder_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # Ignore classification head weights
            if 'head' in k:
                continue
            # Some implementations may have different prefixes
            if k.startswith('backbone.'):
                k = k[9:]  # Remove "backbone." prefix
            elif k.startswith('encoder.'):
                k = k[8:]  # Remove "encoder." prefix

            # Ensure dimensions match
            if k in self.model.state_dict() and \
               v.shape == self.model.state_dict()[k].shape:
                encoder_state_dict[k] = v

        # Load filtered weights
        msg = self.model.load_state_dict(encoder_state_dict, strict=False)
        print(f"Weights loaded. Missing: {len(msg.missing_keys)}, "
              f"Unexpected: {len(msg.unexpected_keys)}")

    def freeze(self):
        """Freezes all encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_gradually(self, stage):
        """
        Gradually unfreezes the encoder in stages.

        Args:
            stage:
              0: Everything frozen
              1: Only patch_embed and pos_drop unfrozen
              2: Stage 1 + final layer (layers[-1]) unfrozen
              3: Stage 2 + layer 3 (layers[-2]) unfrozen
              4: Stage 3 + layer 2 (layers[-3]) unfrozen
              5: Stage 4 + layer 1 (layers[-4]) unfrozen (if it exists)
              6: Everything unfrozen
        """
        # First freeze everything
        self.freeze()

        # Then unfreeze according to the stage
        num_layers = len(self.model.layers)
        max_stage = num_layers + 2  # +2 for patch_embed and everything

        if not 0 <= stage <= max_stage:
            raise ValueError(f"Invalid stage {stage}. "
                             f"Must be between 0 and {max_stage}")

        # Unfreeze components based on stage
        if stage >= 1:  # Unfreeze initial embedding
            for param in self.model.patch_embed.parameters():
                param.requires_grad = True
            for param in self.model.pos_drop.parameters():
                param.requires_grad = True
            if self.model.ape and self.model.absolute_pos_embed is not None:
                self.model.absolute_pos_embed.requires_grad = True

        # Unfreeze layers in reverse order (from highest to lowest)
        for i in range(num_layers):
            if stage >= i + 2:  # +2 because we start at stage 2 for layers
                layer_idx = num_layers - i - 1  # Reverse index
                for param in self.model.layers[layer_idx].parameters():
                    param.requires_grad = True
                print(f"Unfrozen layer {layer_idx + 1} of the encoder")

        # Unfreeze final normalization if we're at the last stage
        if stage >= max_stage:
            for param in self.model.norm.parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        Forward pass that returns intermediate features for skip connections.
        """
        features = []

        # Patch partition
        x = self.model.patch_embed(x)
        if self.model.ape:
            x = x + self.model.absolute_pos_embed
        x = self.model.pos_drop(x)

        # Save features after embedding
        features.append(x)

        # Pass through encoder blocks and save intermediate features
        for i, layer in enumerate(self.model.layers):
            x = layer(x)
            if i < len(self.model.layers) - 1:
                features.append(x)

        # Apply final normalization
        x = self.model.norm()

        # Reverse the list for use in the decoder (deepest first)
        features = features[::-1]

        return x, features


if __name__ == "__main__":
    # Example usage with transfer learning
    model = SwinUNetV2(
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=1,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        pretrained=True,
        pretrained_model_path="saved_models/swinv2_small_window16_256_in1k.\
pth",
        freeze_encoder=True,
        decoder_dropout=0.1
    )

    # Show model structure and state
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Encoder frozen: {not any(p.requires_grad for p in
                                     model.encoder.model.parameters())}")

    # Test unfreezing
    model.unfreeze_next_stage_if_needed(current_metric=0.5, patience=1)
    print(f"Current stage: {model.current_unfreeze_stage}")

    # Simulate several epochs without improvement to test unfreezing
    model.unfreeze_next_stage_if_needed(current_metric=0.5, patience=1)
    print(f"Current stage: {model.current_unfreeze_stage}")
