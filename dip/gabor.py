import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


class GaborFilterBank:
    """
    Implements a bank of Gabor filters for asphalt crack detection.
    """
    def __init__(
        self,
        orientations=np.array([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180]
                              ) * np.pi/180,
        wavelengths=np.array([8, 16, 24]),
        sigma_factor=0.56,
        gamma=1.0,
        psi=0,
        kernel_size=None
    ):
        """
        Initialize a bank of Gabor filters with configurable parameters.

        Parameters:
        -----------
        orientations : array_like
            Filter orientations in radians
        wavelengths : array_like
            Wavelengths of the sinusoidal component
        sigma_factor : float
            Factor relating sigma to wavelength
            (sigma = sigma_factor * wavelength)
        gamma : float
            Spatial aspect ratio
        psi : float
            Phase offset
            (0 for symmetric/cosine filter, π/2 for antisymmetric/sine filter)
        kernel_size : int or None
            Size of the Gabor kernel. If None, it will be calculated as
            6 * sigma
        """
        self.orientations = orientations
        self.wavelengths = wavelengths
        self.sigma_factor = sigma_factor
        self.gamma = gamma
        self.psi = psi
        self.kernel_size = kernel_size
        self.filters = self._create_filter_bank()

    def _create_filter_bank(self):
        """
        Create a bank of Gabor filters with the specified parameters.

        Returns:
        --------
        list of tuples
            Each tuple contains (kernel, orientation, wavelength)
        """
        filters = []

        for theta in self.orientations:
            for lambd in self.wavelengths:
                # Calculate sigma based on wavelength
                sigma = self.sigma_factor * lambd

                # Calculate kernel size if not specified
                ksize = self.kernel_size
                if ksize is None:
                    ksize = int(np.ceil(6 * sigma))
                    # Ensure kernel size is odd
                    ksize = ksize + 1 if ksize % 2 == 0 else ksize

                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    (ksize, ksize),
                    sigma,
                    theta,
                    lambd,
                    self.gamma,
                    self.psi,
                    ktype=cv2.CV_32F
                )

                # Normalize the kernel for better visualization
                kernel /= 1.5*kernel.sum()

                filters.append((kernel, theta, lambd, sigma))

        return filters

    def apply_filters(self, image, normalize=True):
        """
        Apply all Gabor filters to an input image.

        Parameters:
        -----------
        image : ndarray
            Input grayscale image
        normalize : bool
            Whether to normalize output images to 0-255 range

        Returns:
        --------
        list of tuples
            Each tuple contains (filtered_image, orientation, wavelength, sigma
            )
        """
        responses = []

        for kernel, theta, lambd, sigma in self.filters:
            # Apply filter
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)

            # Take absolute value to get magnitude response
            magnitude = np.abs(filtered)

            # Normalize to 0-255 range if requested
            if normalize:
                magnitude = cv2.normalize(magnitude, None, 0, 255,
                                          cv2.NORM_MINMAX).astype(np.uint8)

            responses.append((magnitude, theta, lambd, sigma))

        return responses

    def visualize_kernels(self, figsize=(15, 10)):
        """
        Visualize all Gabor kernels in the filter bank.
        """
        n_kernels = len(self.filters)
        n_cols = min(4, n_kernels)
        n_rows = int(np.ceil(n_kernels / n_cols))

        plt.figure(figsize=figsize)

        for i, (kernel, theta, lambd, sigma) in enumerate(self.filters):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(kernel, cmap='gray')
            theta_deg = np.rad2deg(theta)
            plt.title(f'θ={theta_deg:.1f}°, λ={lambd}, σ={sigma:.1f}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_responses(self, responses, figsize=(18, 12)):
        """
        Visualize the filter responses.

        Parameters:
        -----------
        responses : list
            List of tuples (filtered_image, orientation, wavelength, sigma)
        figsize : tuple
            Figure size
        """
        n_responses = len(responses)
        n_cols = min(4, n_responses)
        n_rows = int(np.ceil(n_responses / n_cols))

        plt.figure(figsize=figsize)

        for i, (response, theta, lambd, sigma) in enumerate(responses):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(response, cmap='gray')
            theta_deg = np.rad2deg(theta)
            plt.title(f'θ={theta_deg:.1f}°, λ={lambd}, σ={sigma:.1f}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def save_responses(self, responses, output_dir, base_filename):
        """
        Save filter responses to disk.

        Parameters:
        -----------
        responses : list
            List of tuples (filtered_image, orientation, wavelength, sigma)
        output_dir : str
            Directory to save images
        base_filename : str
            Base filename without extension
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, (response, theta, lambd, sigma) in enumerate(responses):
            theta_deg = np.rad2deg(theta)
            filename = f"{base_filename}_gabor_theta{theta_deg:.0f}_\
lambda{lambd:.0f}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, response)

    def get_combined_response(self, responses, method='max'):
        """
        Combine multiple filter responses into a single image.

        Parameters:
        -----------
        responses : list
            List of tuples (filtered_image, orientation, wavelength, sigma)
        method : str
            Method to combine responses: 'max', 'mean', 'sum'

        Returns:
        --------
        ndarray
            Combined response image
        """
        # Extract just the response images
        response_images = [r[0] for r in responses]

        if method == 'max':
            combined = np.max(response_images, axis=0)
        elif method == 'mean':
            combined = np.mean(response_images, axis=0)
        elif method == 'sum':
            combined = np.sum(response_images, axis=0)
        else:
            raise ValueError(f"Unknown combination method: {method}")

        # Normalize to 0-255
        combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX
                                 ).astype(np.uint8)
        return combined


def process_single_image(image_path, output_dir, visualize=True,
                         orientations=np.array([0, 22.5, 45, 67.5, 90, 112.5,
                                                135, 157.5, 180]) * np.pi/180,
                         wavelengths=np.array([8, 16, 24]),
                         sigma_factor=0.56,
                         gamma=1.0,
                         psi=0):
    """
    Process a single image with Gabor filters.

    Parameters:
    -----------
    image_path : str
        Path to input grayscale image
    output_dir : str
        Directory to save output images
    visualize : bool
        Whether to display visualizations
    orientations : array_like
        Filter orientations in radians
    wavelengths : array_like
        Wavelengths of sinusoidal component
    sigma_factor : float
        Factor relating sigma to wavelength
    gamma : float
        Spatial aspect ratio
    psi : float
        Phase offset

    Returns:
    --------
    combined_response : ndarray
        Combined Gabor filter response
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Create Gabor filter bank
    gabor_bank = GaborFilterBank(
        orientations=orientations,
        wavelengths=wavelengths,
        sigma_factor=sigma_factor,
        gamma=gamma,
        psi=psi
    )

    # Visualize kernels if requested
    if visualize:
        gabor_bank.visualize_kernels()

    # Apply filters to image
    responses = gabor_bank.apply_filters(image)

    # Visualize responses if requested
    if visualize:
        gabor_bank.visualize_responses(responses)

    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Save individual responses
    gabor_bank.save_responses(responses, output_dir, base_filename)

    # Generate combined response using max method
    combined_response = gabor_bank.get_combined_response(responses,
                                                         method='max')

    # Save combined response
    combined_path = os.path.join(output_dir,
                                 f"{base_filename}_gabor_combined.png")
    cv2.imwrite(combined_path, combined_response)

    # Show original and combined response if requested
    if visualize:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(combined_response, cmap='gray')
        plt.title("Combined Gabor Response (Max)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return combined_response


def process_directory(input_dir, output_dir, file_pattern="*.jpg",
                      orientations=np.array([0, 22.5, 45, 67.5, 90, 112.5,
                                             135, 157.5, 180]) * np.pi/180,
                      wavelengths=np.array([8, 16, 24]),
                      sigma_factor=0.56,
                      gamma=1.0,
                      psi=0):
    """
    Process all images in a directory with Gabor filters.

    Parameters:
    -----------
    input_dir : str
        Directory containing input images
    output_dir : str
        Directory to save output images
    file_pattern : str
        Pattern to match input files
    orientations, wavelengths, sigma_factor, gamma, psi:
        Parameters for Gabor filters
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all matching files
    input_path = Path(input_dir)
    files = list(input_path.glob(file_pattern))

    if not files:
        print(f"No files matching {file_pattern} found in {input_dir}")
        return

    # Create Gabor filter bank once to reuse
    gabor_bank = GaborFilterBank(
        orientations=orientations,
        wavelengths=wavelengths,
        sigma_factor=sigma_factor,
        gamma=gamma,
        psi=psi
    )

    # Process each file
    for file_path in tqdm(files, desc="Processing images"):
        try:
            # Read image
            image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Could not read {file_path}")
                continue

            # Apply filters
            responses = gabor_bank.apply_filters(image)

            # Get base filename
            base_filename = file_path.stem

            # Save individual responses
            gabor_bank.save_responses(responses, output_dir, base_filename)

            # Generate and save combined response
            combined_response = gabor_bank.get_combined_response(responses,
                                                                 method='max')
            combined_path = os.path.join(output_dir,
                                         f"{base_filename}_gabor_combined.png")
            cv2.imwrite(combined_path, combined_response)

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    print(f"Processed {len(files)} images. Results saved to {output_dir}")


def create_unet_input(image_path, gabor_bank=None, return_channels=True):
    """
    Create a multi-channel input for U-Net by combining the original image
    with Gabor filter responses.

    Parameters:
    -----------
    image_path : str
        Path to input grayscale image
    gabor_bank : GaborFilterBank or None
        Gabor filter bank to use. If None, a default one is created
    return_channels : bool
        Whether to return individual channels (True) or a stacked image (False)

    Returns:
    --------
    channels or stacked_image : list or ndarray
        List of individual channels or stacked multi-channel image
    """
    # Read input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Create Gabor filter bank if not provided
    if gabor_bank is None:
        gabor_bank = GaborFilterBank(
            orientations=np.array([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
                                   180]) * np.pi/180,
            wavelengths=np.array([8, 16, 24]),
            sigma_factor=0.56,
            gamma=1.0,
            psi=0
        )

    # Apply filters to image
    responses = gabor_bank.apply_filters(image, normalize=True)

    # Extract response images
    response_images = [r[0] for r in responses]

    # Original image channel
    channels = [image]

    # Add Gabor response channels
    channels.extend(response_images)

    # Add combined response
    combined = gabor_bank.get_combined_response(responses, method='max')
    channels.append(combined)

    if return_channels:
        return channels
    else:
        # Stack channels to create multi-channel image
        stacked = np.stack(channels, axis=-1)
        return stacked


if __name__ == "__main__":
    # Example usage for a single image
    image_path = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/images/14_2.jpg"
    output_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/gabor_features/"

    # Configure Gabor filter parameters
    orientations = np.array([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180]
                            ) * np.pi/180  # in radians
    wavelengths = np.array([8, 16, 24])                   # 3 wavelengths
    sigma_factor = 0.56                          # Relates sigma to wavelength
    gamma = 0.5                                  # Aspect ratio
    psi = 0                                      # Phase offset (0=symmetric)

    # Process image with Gabor filters
    process_single_image(
        image_path=image_path,
        output_dir=output_dir,
        visualize=True,
        orientations=orientations,
        wavelengths=wavelengths,
        sigma_factor=sigma_factor,
        gamma=gamma,
        psi=psi
    )

    # Example for batch processing
    """
    process_directory(
        input_dir="C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
            CrackDataset/luz_crack/images/",
        output_dir="C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
            doctorado/CrackDataset/luz_crack/gabor_features/",
        file_pattern="*.jpg",
        orientations=orientations,
        wavelengths=wavelengths,
        sigma_factor=sigma_factor,
        gamma=gamma,
        psi=psi
    )
    """
