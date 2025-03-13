import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def image_svd_reconstruction(image, k, convert_to_grayscale=False):
    """
    Perform SVD on an image and reconstruct it using only the first k singular
    values.

    Parameters:
        image: Input image (path string or numpy array)
        k: Number of singular values to keep for reconstruction
        convert_to_grayscale: If True, converts color images to grayscale
            before SVD

    Returns:
        reconstructed_image: Image reconstructed with k singular values
    """
    # Handle different input types
    if isinstance(image, str):
        # Load image from path
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not load image from {image}")
        # Convert from BGR to RGB if loaded with OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Assume it's already a numpy array
        img = image.copy()

    # Convert to grayscale if requested
    if convert_to_grayscale and len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Get image dimensions and check if grayscale or color
    if len(img.shape) == 2:
        # Grayscale image
        return svd_reconstruct_channel(img, k)
    else:
        # Color image - process each channel separately
        height, width, channels = img.shape
        reconstructed = np.zeros_like(img, dtype=np.float32)

        for c in range(channels):
            reconstructed[:, :, c] = svd_reconstruct_channel(img[:, :, c], k)

        # Ensure pixel values stay within valid range
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        return reconstructed


def svd_reconstruct_channel(channel, k):
    """
    Apply SVD to a single channel and reconstruct using k components.

    Parameters:
        channel: 2D array representing image channel
        k: Number of singular values to keep

    Returns:
        reconstructed_channel: Reconstructed 2D array
    """
    # Convert to float for better precision
    channel_float = channel.astype(np.float32)

    # Apply SVD
    U, sigma, Vt = np.linalg.svd(channel_float, full_matrices=False)

    # Limit k to the maximum possible number of singular values
    k = min(k, len(sigma))

    # Reconstruct using only k singular values
    reconstructed = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]

    return reconstructed


def visualize_svd_reconstruction(image, k_values, convert_to_grayscale=False):
    """
    Visualize original image and its reconstructions with different k values.

    Parameters:
        image: Input image (path or array)
        k_values: List of k values to use for reconstruction
        convert_to_grayscale: If True, converts color images to grayscale
            before SVD
    """
    # Load image if it's a path
    if isinstance(image, str):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image.copy()

    # Convert to grayscale if requested
    if convert_to_grayscale and len(img.shape) > 2:
        img_to_display = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cmap = 'gray'
    else:
        img_to_display = img
        cmap = None

    # Create a figure with subplots
    n_plots = len(k_values) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))

    # Plot original image
    axes[0].imshow(img_to_display, cmap=cmap)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Plot reconstructed images with different k values
    for i, k in enumerate(k_values):
        reconstructed = image_svd_reconstruction(img, k, convert_to_grayscale)
        axes[i+1].imshow(reconstructed, cmap=cmap if convert_to_grayscale else
                         None)
        axes[i+1].set_title(f"k = {k}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()


def calculate_compression_ratio(image, k, convert_to_grayscale=False):
    """
    Calculate the compression ratio achieved by SVD with k components.

    Parameters:
        image: Input image (path or array)
        k: Number of singular values used
        convert_to_grayscale: If True, converts color images to grayscale
            before SVD

    Returns:
        ratio: Compression ratio
    """
    # Get image dimensions
    if isinstance(image, str):
        img = cv2.imread(image)
        if convert_to_grayscale and len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        channels = 1 if len(img.shape) == 2 or convert_to_grayscale else 3
    else:
        if convert_to_grayscale and len(image.shape) > 2:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            h, w = img.shape
            channels = 1
        else:
            h, w = image.shape[:2]
            channels = 1 if len(image.shape) == 2 else 3

    # Calculate sizes
    original_size = h * w * channels

    # In SVD, we store:
    # - U: h × k values per channel
    # - Sigma: k values per channel
    # - V^T: k × w values per channel
    svd_size = (h*k + k + k*w) * channels

    return original_size / svd_size


if __name__ == "__main__":
    # Example usage
    image_path = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/images/14_2.jpg"
    k = 10  # Number of singular values to keep
    use_grayscale = True  # Set to True to process in grayscale

    # Reconstruct the image
    reconstructed = image_svd_reconstruction(image_path, k, use_grayscale)

    # Display the results
    visualize_svd_reconstruction(image_path, [1, 5, 10], use_grayscale)

    # Calculate and print compression ratio
    ratio = calculate_compression_ratio(image_path, k, use_grayscale)
    print(f"Compression ratio with k={k}: {ratio:.2f}x")

    # Save the reconstructed image
    result_img = Image.fromarray(reconstructed)
    mode = "L" if use_grayscale else "RGB"
    result_img = Image.fromarray(reconstructed, mode=mode)
    result_img.save(f"reconstructed_k{k}_\
{'gray' if use_grayscale else 'color'}.png")
    print(f"Reconstructed image saved as reconstructed_k{k}_\
{'gray' if use_grayscale else 'color'}.png")
