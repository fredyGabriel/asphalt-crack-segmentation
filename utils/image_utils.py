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

    # Convert to numpy array for processing
    if is_pil:
        img_array = np.array(image.convert('L')) / 255.0
    elif is_tensor:
        img_array = image.cpu().numpy()
        # If it's a multi-channel tensor, convert to grayscale
        if len(img_array.shape) == 3 and img_array.shape[0] > 1:
            img_array = 0.299 * img_array[0] + 0.587 * img_array[1] + 0.114 *\
                img_array[2]
    else:
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            # Convert RGB to grayscale
            img_array = 0.299 * img_array[:, :, 0] + 0.587 *\
                img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        img_array = img_array / 255.0 if img_array.max() > 1.0 else img_array

    # Binarize
    binary_array = (img_array > threshold).astype(np.float32)

    # Invert if necessary
    if invert:
        binary_array = 1.0 - binary_array

    # Return in the original format
    if is_pil:
        return Image.fromarray((binary_array * 255).astype(np.uint8))
    elif is_tensor:
        if len(image.shape) == 3 and image.shape[0] == 1:
            # Single-channel tensor
            return torch.from_numpy(binary_array).unsqueeze(0).to(image.device)
        elif len(image.shape) == 3:
            # Multi-channel tensor
            return torch.from_numpy(binary_array).to(image.device)
        else:
            return torch.from_numpy(binary_array).to(image.device)
    else:
        return binary_array


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

    # Convert to numpy array
    img_array = np.array(img_gray)

    height, width = img_array.shape
    result = np.zeros_like(img_array, dtype=np.uint8)

    # Block processing
    pad = block_size // 2
    padded_img = np.pad(img_array, pad, mode='reflect')

    for i in range(height):
        for j in range(width):
            # Extract block
            block = padded_img[i:i+block_size, j:j+block_size]
            # Calculate threshold
            threshold = np.mean(block) - C
            # Apply threshold
            result[i, j] = 255 if img_array[i, j] > threshold else 0

    return Image.fromarray(result)


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
        # Manual implementation of Otsu
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert('L'))
        else:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                # Convert RGB to grayscale
                img_array = 0.299 * img_array[:, :, 0] + 0.587 *\
                    img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
                img_array = img_array.astype(np.uint8)

        # Calculate histogram
        hist, bin_edges = np.histogram(img_array, bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()

        # Variables for optimal threshold
        best_thresh = 0
        best_variance = 0

        # Calculate between-class variance for each possible threshold
        for t in range(1, 255):
            # Weights
            w0 = np.sum(hist[:t])
            w1 = 1 - w0

            if w0 == 0 or w1 == 0:
                continue

            # Means
            mu0 = np.sum(np.arange(t) * hist[:t]) / w0
            mu1 = np.sum(np.arange(t, 256) * hist[t:]) / w1

            # Between-class variance
            variance = w0 * w1 * (mu0 - mu1) ** 2

            # Update if we find better variance
            if variance > best_variance:
                best_variance = variance
                best_thresh = t

        # Apply optimal threshold
        binary = np.where(img_array > best_thresh, 255, 0).astype(np.uint8)
        return Image.fromarray(binary)
