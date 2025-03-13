import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Se implementa la clase DoubleConv con la posibilidad de añadir dropout
    después de cada bloque de convolución."""
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        # Añado la posibilidad de incluir dropout después de cada bloque de
        # convolución
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        # Añadir dropout después del primer bloque si se especifica
        if dropout_p > 0:
            layers.append(nn.Dropout2d(p=dropout_p))

        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])

        # Añadir dropout después del segundo bloque si se especifica
        if dropout_p > 0:
            layers.append(nn.Dropout2d(p=dropout_p))

        self.conv_op = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, dropout_p=dropout_p)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2,
                                     kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match x2's dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1,
                 encoder_dropout=0.1, bottleneck_dropout=0.5,
                 decoder_dropout=0.1):
        """
        UNet con dropout configurable en diferentes secciones.

        Args:
            in_channels: Número de canales de entrada (3 para RGB)
            out_channels: Número de canales de salida (1 para segmentación
                binaria)
            encoder_dropout: Tasa de dropout para el codificador (0.0-1.0)
            bottleneck_dropout: Tasa de dropout para el cuello de botella
                (0.0-1.0)
            decoder_dropout: Tasa de dropout para el decodificador (0.0-1.0)
        """
        super().__init__()

        # Codificador con dropout variable y creciente en profundidad
        self.down_convolution_1 = DownSample(in_channels, 64,
                                             dropout_p=encoder_dropout * 1.0)
        self.down_convolution_2 = DownSample(64, 128,
                                             dropout_p=encoder_dropout * 2.0)
        self.down_convolution_3 = DownSample(128, 256,
                                             dropout_p=encoder_dropout * 3.0)
        self.down_convolution_4 = DownSample(256, 512,
                                             dropout_p=encoder_dropout * 4.0)

        # Cuello de botella con el mayor dropout
        self.bottle_neck = DoubleConv(512, 1024, dropout_p=bottleneck_dropout)

        # Decodificador con dropout decreciente
        self.up_convolution_1 = UpSample(1024, 512,
                                         dropout_p=decoder_dropout * 4.0)
        self.up_convolution_2 = UpSample(512, 256,
                                         dropout_p=decoder_dropout * 3.0)
        self.up_convolution_3 = UpSample(256, 128,
                                         dropout_p=decoder_dropout * 2.0)
        self.up_convolution_4 = UpSample(128, 64,
                                         dropout_p=decoder_dropout * 1.0)

        # No aplicamos dropout antes de la salida final
        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels,
                             kernel_size=1)

    def forward(self, x):
        # Codificador
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        # Cuello de botella
        b = self.bottle_neck(p4)

        # Decodificador
        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        # Capa de salida
        out = self.out(up_4)
        return out


if __name__ == "__main__":
    # Probamos el modelo con diferentes tasas de dropout
    model = UNet2(
        in_channels=3,
        out_channels=1,
        encoder_dropout=0.1,
        bottleneck_dropout=0.5,
        decoder_dropout=0.1
    )
    x = torch.randn((1, 3, 256, 256))
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Contar parámetros del modelo
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
