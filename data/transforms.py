"""
Transformations for image segmentation tasks.

This module provides transformation classes specifically designed for
simultaneously processing both input images and their corresponding
segmentation masks. All transformations maintain the correct correspondence
between images and masks.
"""

import torch
import math
import numpy as np
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F
from typing import List, Tuple, Callable, Union


class Compose:
    """
    Composes several transforms together to be applied to image and mask pairs.

    This class ensures that the same transformations are applied to both
    the input image and its corresponding segmentation mask, maintaining their
        alignment.
    """

    def __init__(self, transforms: List[Callable]) -> None:
        """
        Initialize the compose transformation with a list of transforms.

        Args:
            transforms: List of transformations to compose.
        """
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image) ->\
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply all transformations sequentially to the image and mask.

        Args:
            image: Input image to transform
            mask: Corresponding segmentation mask

        Returns:
            Tuple containing the transformed image and mask
        """
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class ToTensor:
    """
    Convert PIL images to PyTorch tensors.

    Converts image to a tensor with values in range [0, 1] and
    converts mask to a binary tensor with values 0 or 1.
    """

    def __call__(self, image: Image.Image, mask: Image.Image) ->\
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert image and mask from PIL images to PyTorch tensors.

        Args:
            image: Input image as PIL Image
            mask: Corresponding segmentation mask as PIL Image

        Returns:
            Tuple containing the image as tensor with shape (C, H, W)
            and mask as tensor with shape (1, H, W)
        """
        # Convert PIL images to tensors
        image = transforms.ToTensor()(image)
        # For masks, ensure binary values (0 or 1)
        mask = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0)
        if len(mask.shape) == 2:  # Add channel dim if needed
            mask = mask.unsqueeze(0)
        return image, mask


class Normalize:
    """
    Normalize image tensor with mean and standard deviation.

    This transformation normalizes only the image tensor, leaving the mask
    unchanged.
    """

    def __init__(self, mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225]) -> None:
        """
        Initialize the normalization transformation.

        Args:
            mean: Mean values for each channel. Default is ImageNet means.
            std: Standard deviation values for each channel. Default is
                ImageNet stds.
        """
        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) ->\
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize the image tensor while keeping mask unchanged.

        Args:
            image: Image tensor with shape (C, H, W)
            mask: Mask tensor with shape (1, H, W)

        Returns:
            Tuple containing the normalized image and unchanged mask
        """
        # Apply normalization only to the image (not the mask)
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask


class RandomHorizontalFlip:
    """
    Horizontally flip the image and mask randomly with a given probability.
    """

    def __init__(self, p: float = 0.5) -> None:
        """
        Initialize the random horizontal flip transformation.

        Args:
            p: Probability of applying the flip. Default is 0.5.
        """
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) ->\
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly flip image and mask horizontally with probability p.

        Args:
            image: Image tensor with shape (C, H, W)
            mask: Mask tensor with shape (1, H, W)

        Returns:
            Tuple containing the possibly flipped image and mask
        """
        if torch.rand(1).item() < self.p:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask


class RandomVerticalFlip:
    """
    Vertically flip the image and mask randomly with a given probability.
    """

    def __init__(self, p: float = 0.5) -> None:
        """
        Initialize the random vertical flip transformation.

        Args:
            p: Probability of applying the flip. Default is 0.5.
        """
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) ->\
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly flip image and mask vertically with probability p.

        Args:
            image: Image tensor with shape (C, H, W)
            mask: Mask tensor with shape (1, H, W)

        Returns:
            Tuple containing the possibly flipped image and mask
        """
        if torch.rand(1).item() < self.p:
            image = F.vflip(image)
            mask = F.vflip(mask)
        return image, mask


class Resize:
    """
    Resize image and mask to a specified size.

    For images, uses bilinear interpolation with antialiasing.
    For masks, uses nearest neighbor interpolation to preserve binary values.
    """

    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        """
        Initialize the resize transformation.

        Args:
            size: Target size as (height, width) tuple or single integer for
                square images
        """
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) ->\
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Resize image and mask to the specified size.

        Args:
            image: Image tensor with shape (C, H, W)
            mask: Mask tensor with shape (1, H, W)

        Returns:
            Tuple containing the resized image and mask
        """
        # Adjust for tensor format (C,H,W)
        image = F.resize(image, size=self.size, antialias=True)
        # For mask, we use nearest interpolation to preserve binary values
        mask = F.resize(mask, size=self.size,
                        interpolation=F.InterpolationMode.NEAREST)
        return image, mask


class RandomNoise:
    """
    Adds realistic noise to simulate various imaging conditions on asphalt
    pavements.
    """

    # Constantes de clase para mejorar legibilidad
    GAUSSIAN = 0
    SALT_PEPPER = 1
    SPECKLE = 2

    def __init__(self, p=0.5, noise_types=None, intensity=0.1):
        """Inicializa la transformación de ruido aleatorio."""
        self.p = p

        # Usa comprensión de lista con filtro (ahora posible sin TorchScript)
        self.noise_types = [0, 1, 2] if noise_types is None else [
            t for t in noise_types if 0 <= t <= 2
        ] or [0, 1, 2]  # Operador or para caso vacío

        self.intensity = max(0.0, min(1.0, intensity))

        # Funciones por tipo de ruido - mejora modularidad y legibilidad
        self.noise_functions = {
            self.GAUSSIAN: self._apply_gaussian_noise,
            self.SALT_PEPPER: self._apply_salt_pepper_noise,
            self.SPECKLE: self._apply_speckle_noise
        }

    def __call__(self, image, mask):
        if torch.rand(1).item() < self.p:
            # Seleccionar tipo de ruido aleatoriamente
            noise_idx = self.noise_types[torch.randint(0,
                                                       len(self.noise_types),
                                                       (1,)).item()]

            # Usar el diccionario de funciones para mejorar legibilidad
            image = self.noise_functions[noise_idx](image)

        return image, mask

    def _apply_gaussian_noise(self, image):
        """Aplica ruido gaussiano a la imagen."""
        std_dev = self.intensity * 0.15
        noise = torch.randn_like(image) * std_dev
        return torch.clamp(image + noise, 0, 1)

    def _apply_salt_pepper_noise(self, image):
        """Aplica ruido de sal y pimienta a la imagen."""
        s_vs_p = 0.5
        amount = self.intensity * 0.05

        # Salt (white pixels)
        salt_mask = torch.rand_like(image) < amount * s_vs_p
        image = torch.where(salt_mask, torch.ones_like(image), image)

        # Pepper (black pixels)
        pepper_mask = torch.rand_like(image) < amount * (1 - s_vs_p)
        image = torch.where(pepper_mask, torch.zeros_like(image), image)

        return image

    def _apply_speckle_noise(self, image):
        """Aplica ruido de manchas a la imagen."""
        noise = torch.randn_like(image) * self.intensity * 0.15 + 1
        return torch.clamp(image * noise, 0, 1)


class RandomShadow:
    """
    Adds realistic shadow patterns that simulate shadows on asphalt pavements.
    """

    def __init__(self, p=0.5, num_shadows_range=(1, 3),
                 shadow_darkness_range=(0.3, 0.7)):
        self.p = p
        self.min_shadows, self.max_shadows = num_shadows_range
        self.min_darkness, self.max_darkness = shadow_darkness_range

        # Pre-calcular valores constantes para evitar recálculos
        self.pi = torch.tensor(math.pi)  # Usar constante en lugar de torch.pi

    def __call__(self, image, mask):
        if torch.rand(1).item() < self.p:
            c, h, w = image.shape
            shadow_map = torch.ones((h, w), device=image.device)

            # Generar múltiples sombras
            n_shadows = torch.randint(self.min_shadows, self.max_shadows + 1,
                                      (1,)).item()

            for _ in range(n_shadows):
                # Generar una sombra y aplicarla al mapa de sombras
                shadow_factor = self._generate_shadow(h, w, image.device)
                shadow_map = shadow_map * shadow_factor

            # Aplicar el mapa de sombras a la imagen
            shadow_map = shadow_map.unsqueeze(0).expand_as(image)
            image = image * shadow_map

        return image, mask

    def _generate_shadow(self, h, w, device):
        """Genera una única sombra elíptica con bordes suaves."""
        # Parámetros de la sombra
        center_x = torch.randint(0, w, (1,)).item()
        center_y = torch.randint(0, h, (1,)).item()

        # Dimensiones de la sombra (como porcentajes de las dimensiones de la
        # imagen)
        shadow_width = int(w * (0.2 + 0.6 * torch.rand(1).item()))
        shadow_height = int(h * (0.2 + 0.6 * torch.rand(1).item()))

        # Ángulo de rotación aleatorio
        angle = torch.rand(1).item() * self.pi
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)

        # Oscuridad aleatoria
        darkness = self.min_darkness + torch.rand(1).item() *\
            (self.max_darkness - self.min_darkness)

        # Creamos las coordenadas de forma más eficiente
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # Trasladar coordenadas al origen en el centro de la sombra
        y_trans = y_coords - center_y
        x_trans = x_coords - center_x

        # Aplicar rotación a las coordenadas
        y_rot = y_trans * cos_angle - x_trans * sin_angle
        x_rot = y_trans * sin_angle + x_trans * cos_angle

        # Escalar coordenadas para verificar elipse
        y_scaled = (y_rot / (shadow_height * 0.5)) ** 2
        x_scaled = (x_rot / (shadow_width * 0.5)) ** 2

        # Calcular distancia elíptica (valores <= 1 están dentro de la elipse)
        distance = y_scaled + x_scaled

        # Crear máscara de sombra suave con bordes degradados
        shadow_intensity = torch.clamp(
            1.0 - torch.exp(-2 * torch.abs(distance - 1)), 0, 1)

        # Factor de sombra - más oscuro donde shadow_intensity es menor
        shadow_factor = 1.0 - (darkness * (1.0 - shadow_intensity))

        return shadow_factor
