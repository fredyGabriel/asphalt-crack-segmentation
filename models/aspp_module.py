from torch import nn
import torch


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module for capturing multi-scale features.
    Adapted to work with transformer-based sequence representations.
    """
    def __init__(self, input_dim, output_dim, input_resolution,
                 rates=[2, 4, 6]):
        super(ASPPModule, self).__init__()
        self.input_resolution = input_resolution

        # Convert sequence to feature map format
        self.to_feature_map = nn.Linear(input_dim, output_dim)

        # ASPP branches with different dilation rates
        self.aspp1 = nn.Conv2d(output_dim, output_dim, 1, bias=False)
        self.aspp2 = nn.Conv2d(output_dim, output_dim, 3, padding=rates[0],
                               dilation=rates[0], bias=False)
        self.aspp3 = nn.Conv2d(output_dim, output_dim, 3, padding=rates[1],
                               dilation=rates[1], bias=False)
        self.aspp4 = nn.Conv2d(output_dim, output_dim, 3, padding=rates[2],
                               dilation=rates[2], bias=False)

        # Batch normalization and activation for each branch
        self.norm1 = nn.BatchNorm2d(output_dim)
        self.norm2 = nn.BatchNorm2d(output_dim)
        self.norm3 = nn.BatchNorm2d(output_dim)
        self.norm4 = nn.BatchNorm2d(output_dim)

        # Global pooling branch for capturing global context
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_conv = nn.Conv2d(output_dim, output_dim, 1, bias=False)
        self.global_norm = nn.BatchNorm2d(output_dim)

        # Projection layer after concatenating all branches
        self.projection = nn.Conv2d(output_dim * 5, output_dim, 1, bias=False)
        self.proj_norm = nn.BatchNorm2d(output_dim)

        # Activation function
        self.act = nn.GELU()

        # Convert back to sequence format
        self.to_sequence = nn.Linear(output_dim, input_dim)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution

        # Verify that the sequence length matches the expected resolution
        assert L == H * W, f"Input feature size {L} doesn't match resolution \
{H}x{W}"

        # Project features to the desired output dimension
        x = self.to_feature_map(x)

        # Reshape to 2D feature map format [B, C, H, W]
        x_img = x.transpose(1, 2).reshape(B, -1, H, W)

        # Apply ASPP branches
        x1 = self.act(self.norm1(self.aspp1(x_img)))
        x2 = self.act(self.norm2(self.aspp2(x_img)))
        x3 = self.act(self.norm3(self.aspp3(x_img)))
        x4 = self.act(self.norm4(self.aspp4(x_img)))

        # Global pooling branch
        x5 = self.global_avg_pool(x_img)
        x5 = self.act(self.global_norm(self.global_conv(x5)))
        x5 = torch.nn.functional.interpolate(x5, size=(H, W), mode='bilinear',
                                             align_corners=True)

        # Concatenate all branches along channel dimension
        x_cat = torch.cat((x1, x2, x3, x4, x5), dim=1)

        # Project back to output dimension
        x_proj = self.act(self.proj_norm(self.projection(x_cat)))

        # Reshape back to sequence format [B, H*W, C]
        x_seq = x_proj.flatten(2).transpose(1, 2)

        # Project to original dimension
        x_out = self.to_sequence(x_seq)

        return x_out
