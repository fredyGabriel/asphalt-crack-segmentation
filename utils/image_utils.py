from PIL import Image
import os
import numpy as np
import torch
from typing import Tuple, Optional, Union


def load_image(image_path: str) -> Image.Image:
    """
    Loads an image from a file path.

    Args:
        image_path: Path to image file

    Returns:
        Loaded image as PIL.Image object

    Raises:
        FileNotFoundError: If the image doesn't exist
        IOError: If there's an error loading the image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    try:
        return Image.open(image_path)
    except IOError as e:
        raise IOError(f"Error loading image: {e}")


def load_image_and_mask(image_path: str, mask_path: str) -> Tuple[Image.Image,
                                                                  Image.Image]:
    """
    Loads an image and its corresponding mask.

    Args:
        image_path: Path to image file
        mask_path: Path to mask file

    Returns:
        Tuple with the image and mask as PIL.Image objects
    """
    image = load_image(image_path)
    mask = load_image(mask_path)

    return image, mask


def save_image(image: Image.Image, save_path: str,
               format: Optional[str] = None) -> None:
    """
    Saves an image to a specific path.

    Args:
        image: Image to save
        save_path: Path where to save the image
        format: Image format (optional, inferred from extension if
            not provided)
    """
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    image.save(save_path, format=format)


def binarize_image(image: Union[Image.Image, np.ndarray, torch.Tensor],
                   threshold: float = 0.5,
                   invert: bool = False) -> Union[Image.Image, np.ndarray,
                                                  torch.Tensor]:
    """
    Binarizes an image using a specific threshold.

    Args:
        image: Image to binarize (PIL, numpy array or PyTorch tensor)
        threshold: Threshold value for binarization (between 0 and 1)
        invert: If True, inverts the values after binarization

    Returns:
        Binarized image in the same format as the input
    """
    # Determine image type
    is_pil = isinstance(image, Image.Image)
    is_tensor = isinstance(image, torch.Tensor)
    device = torch.device('cuda' if torch.cuda.is_available() and is_tensor
                          and image.is_cuda else 'cpu')

    # Convert to PyTorch tensor for processing
    if is_pil:
        img_array = np.array(image.convert('L')) / 255.0
        img_tensor = torch.tensor(img_array, dtype=torch.float32,
                                  device=device)
    elif is_tensor:
        img_tensor = image
        # If it's a multi-channel tensor, convert to grayscale
        if len(img_tensor.shape) == 3 and img_tensor.shape[0] > 1:
            img_tensor = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] +\
                0.114 * img_tensor[2]
    else:
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            # Convert RGB to grayscale
            img_array = 0.299 * img_array[:, :, 0] +\
                0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        img_array = img_array / 255.0 if img_array.max() > 1.0 else img_array
        img_tensor = torch.tensor(img_array, dtype=torch.float32,
                                  device=device)

    # Binarize using PyTorch operations
    binary_tensor = (img_tensor > threshold).float()

    # Invert if necessary
    if invert:
        binary_tensor = 1.0 - binary_tensor

    # Return in the original format
    if is_pil:
        return Image.fromarray((binary_tensor.cpu().numpy() * 255).
                               astype(np.uint8))
    elif is_tensor:
        if len(image.shape) == 3 and image.shape[0] == 1:
            # Single-channel tensor
            return binary_tensor.unsqueeze(0).to(image.device)
        elif len(image.shape) == 3:
            # Multi-channel tensor
            return binary_tensor.to(image.device)
        else:
            return binary_tensor.to(image.device)
    else:
        return binary_tensor.cpu().numpy()


def adaptive_threshold(image: Union[Image.Image, np.ndarray],
                       block_size: int = 11,
                       C: float = 2.0) -> Image.Image:
    """
    Applies adaptive thresholding to an image.

    Uses a different threshold for each small region of the image.

    Args:
        image: Input image
        block_size: Size of the window for calculating local threshold
        C: Constant subtracted from the block average

    Returns:
        Binarized image with adaptive threshold
    """
    # Convert to PIL if it's a numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Convert to grayscale
    img_gray = image.convert('L')

    # Convert to PyTorch tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = torch.tensor(list(img_gray.getdata()), dtype=torch.float32,
                              device=device)
    img_tensor = img_tensor.reshape(img_gray.height, img_gray.width)

    height, width = img_tensor.shape
    result = torch.zeros_like(img_tensor)

    # Block processing
    pad = block_size // 2
    padded_img = torch.nn.functional.pad(img_tensor, (pad, pad, pad, pad),
                                         mode='reflect')

    # Using PyTorch unfold for sliding window
    patches = padded_img.unfold(0, block_size, 1).unfold(1, block_size, 1)

    # Calculate thresholds for each block
    thresholds = torch.mean(patches, dim=(2, 3)) - C

    # Apply thresholds
    result = torch.where(img_tensor > thresholds,
                         torch.tensor(255.0, device=device),
                         torch.tensor(0.0, device=device))

    return Image.fromarray(result.cpu().byte().numpy())


def otsu_threshold(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    """
    Applies the Otsu thresholding method.

    Automatically calculates the optimal threshold based on the
    histogram distribution of the image.

    Args:
        image: Input image

    Returns:
        Binarized image with Otsu threshold
    """
    try:
        # If OpenCV is available, use it for Otsu
        import cv2

        # Convert to format compatible with OpenCV
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert('L'))
        else:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply Otsu thresholding
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY +
                                  cv2.THRESH_OTSU)
        return Image.fromarray(binary)

    except ImportError:
        # Manual implementation of Otsu using PyTorch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if isinstance(image, Image.Image):
            img_array = np.array(image.convert('L'))
        else:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                # Convert RGB to grayscale
                img_array = 0.299 * img_array[:, :, 0] +\
                    0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
                img_array = img_array.astype(np.uint8)

        img_tensor = torch.tensor(img_array, dtype=torch.float32,
                                  device=device)

        # Calculate histogram
        hist = torch.histc(img_tensor, bins=256, min=0, max=255)
        hist = hist / hist.sum()

        # Variables for optimal threshold
        best_thresh = 0
        best_variance = 0

        # Create a range tensor for calculations
        range_tensor = torch.arange(256, dtype=torch.float32, device=device)

        # Calculate between-class variance for each possible threshold
        for t in range(1, 255):
            # Weights
            w0 = torch.sum(hist[:t])
            w1 = 1 - w0

            if w0 == 0 or w1 == 0:
                continue

            # Means
            mu0 = torch.sum(range_tensor[:t] * hist[:t]) / w0
            mu1 = torch.sum(range_tensor[t:] * hist[t:]) / w1

            # Between-class variance
            variance = w0 * w1 * (mu0 - mu1) ** 2

            # Update if we find better variance
            if variance > best_variance:
                best_variance = variance
                best_thresh = t

        # Apply optimal threshold
        binary = torch.where(img_tensor > best_thresh,
                             torch.tensor(255.0, device=device),
                             torch.tensor(0.0, device=device))
        return Image.fromarray(binary.cpu().byte().numpy())
