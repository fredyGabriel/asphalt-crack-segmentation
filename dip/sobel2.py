import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


class SobelProcessor:
    """
    Class for applying Sobel operators to detect edges in images with
    customizable outputs including gradients, magnitude, and orientation.
    """
    def __init__(
        self,
        ksize=3,
        scale=1,
        delta=0,
        normalize_output=True,
        border_type=cv2.BORDER_DEFAULT
    ):
        """
        Initialize the Sobel processor with configurable parameters.

        Parameters:
        -----------
        ksize : int
            Size of the Sobel kernel (1, 3, 5, or 7)
        scale : float
            Scale factor for computed derivatives
        delta : float
            Value added to results (typically 0)
        normalize_output : bool
            Whether to normalize outputs to 0-255 range
        border_type : int
            OpenCV border type for filter operations
        """
        self.ksize = ksize
        self.scale = scale
        self.delta = delta
        self.normalize_output = normalize_output
        self.border_type = border_type

    def calculate_gradients(self, image):
        """
        Calculate the Sobel gradients in both x and y directions.

        Parameters:
        -----------
        image : ndarray
            Input grayscale image

        Returns:
        --------
        tuple
            (gradient_x, gradient_y) - raw gradient values
        """
        # Ensure image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Sobel operator in x direction
        grad_x = cv2.Sobel(
            image,
            cv2.CV_32F,
            1, 0,
            ksize=self.ksize,
            scale=self.scale,
            delta=self.delta,
            borderType=self.border_type
        )

        # Apply Sobel operator in y direction
        grad_y = cv2.Sobel(
            image,
            cv2.CV_32F,
            0, 1,
            ksize=self.ksize,
            scale=self.scale,
            delta=self.delta,
            borderType=self.border_type
        )

        return grad_x, grad_y

    def calculate_edge_strength(self, grad_x, grad_y):
        """
        Calculate the edge strength (magnitude of gradient).

        Parameters:
        -----------
        grad_x, grad_y : ndarray
            Gradient values in x and y directions

        Returns:
        --------
        ndarray
            Edge strength (magnitude)
        """
        # Calculate magnitude using L2 norm (sqrt(x^2 + y^2))
        magnitude = cv2.magnitude(grad_x, grad_y)

        return magnitude

    def calculate_edge_orientation(self, grad_x, grad_y):
        """
        Calculate the edge orientation (angle of gradient).

        Parameters:
        -----------
        grad_x, grad_y : ndarray
            Gradient values in x and y directions

        Returns:
        --------
        ndarray
            Edge orientation in radians (-π to π)
        """
        # Calculate orientation (atan2(y, x))
        orientation = cv2.phase(grad_x, grad_y)

        return orientation

    def normalize_image(self, image):
        """
        Normalize image to 0-255 range for visualization.

        Parameters:
        -----------
        image : ndarray
            Input image

        Returns:
        --------
        ndarray
            Normalized image (uint8)
        """
        if self.normalize_output:
            return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX
                                 ).astype(np.uint8)
        return image

    def process_image(self, image, calc_gradient_x=True, calc_gradient_y=True,
                      calc_edge_strength=True, calc_edge_orientation=True):
        """
        Process an image with Sobel operators and return selected outputs.

        Parameters:
        -----------
        image : ndarray
            Input image (grayscale or color)
        calc_gradient_x : bool
            Whether to calculate and return x gradient
        calc_gradient_y : bool
            Whether to calculate and return y gradient
        calc_edge_strength : bool
            Whether to calculate and return edge strength
        calc_edge_orientation : bool
            Whether to calculate and return edge orientation

        Returns:
        --------
        dict
            Dictionary with the requested outputs
        """
        # Calculate the basic gradients
        grad_x, grad_y = self.calculate_gradients(image)

        # Initialize results dictionary
        results = {}

        # Add requested outputs to results
        if calc_gradient_x:
            results['gradient_x'] = self.normalize_image(grad_x)

        if calc_gradient_y:
            results['gradient_y'] = self.normalize_image(grad_y)

        if calc_edge_strength:
            magnitude = self.calculate_edge_strength(grad_x, grad_y)
            results['edge_strength'] = self.normalize_image(magnitude)

        if calc_edge_orientation:
            orientation = self.calculate_edge_orientation(grad_x, grad_y)
            # Convert radians to degrees for easier interpretation
            orientation_deg = np.rad2deg(orientation)
            # Normalize to 0-255 for visualization
            results['edge_orientation'] = self.normalize_image(orientation_deg)
            # Store the raw orientation data for potential further use
            results['edge_orientation_raw'] = orientation

        return results


def visualize_sobel_results(image, results, figsize=(15, 10)):
    """
    Visualize original image and selected Sobel results.

    Parameters:
    -----------
    image : ndarray
        Original input image
    results : dict
        Dictionary with Sobel processing results
    figsize : tuple
        Figure size for the plot
    """
    # Count total number of plots (original + results)
    n_plots = 1 + len(results)

    # Calculate grid layout
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    plt.figure(figsize=figsize)

    # Plot original image
    plt.subplot(n_rows, n_cols, 1)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Plot results
    idx = 2
    for title, result in results.items():
        if title.endswith('_raw'):  # Skip raw data
            continue

        plt.subplot(n_rows, n_cols, idx)

        # Choose colormap based on result type
        if title == 'edge_orientation':
            # Use circular colormap for orientation
            plt.imshow(result, cmap='hsv')
        else:
            plt.imshow(result, cmap='viridis')

        plt.title(title.replace('_', ' ').title())
        plt.axis('off')
        idx += 1

    plt.tight_layout()
    plt.show()


def save_sobel_results(output_dir, base_filename, results):
    """
    Save Sobel results to disk.

    Parameters:
    -----------
    output_dir : str
        Directory to save results
    base_filename : str
        Base filename without extension
    results : dict
        Dictionary with Sobel processing results
    """
    os.makedirs(output_dir, exist_ok=True)

    for title, result in results.items():
        if title.endswith('_raw'):  # Skip raw data
            continue

        filename = f"{base_filename}_sobel_{title}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, result)

    print(f"Saved {len(results) - (1 if 'edge_orientation_raw' in results else
                                   0)} result images to {output_dir}")


def process_single_image(image_path, output_dir=None, visualize=True,
                         calc_gradient_x=True, calc_gradient_y=True,
                         calc_edge_strength=True, calc_edge_orientation=True,
                         ksize=3):
    """
    Process a single image with Sobel operators.

    Parameters:
    -----------
    image_path : str
        Path to input image
    output_dir : str or None
        Directory to save outputs (None for no saving)
    visualize : bool
        Whether to display visualizations
    calc_gradient_x, calc_gradient_y, calc_edge_strength,
        calc_edge_orientation : bool
        Options to control which outputs are calculated
    ksize : int
        Size of the Sobel kernel

    Returns:
    --------
    dict
        Dictionary with the requested outputs
    """
    # Read input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create Sobel processor
    processor = SobelProcessor(ksize=ksize)

    # Process the image
    results = processor.process_image(
        gray,
        calc_gradient_x=calc_gradient_x,
        calc_gradient_y=calc_gradient_y,
        calc_edge_strength=calc_edge_strength,
        calc_edge_orientation=calc_edge_orientation
    )

    # Save results if output directory is provided
    if output_dir:
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        save_sobel_results(output_dir, base_filename, results)

    # Visualize if requested
    if visualize:
        visualize_sobel_results(gray, results)

    return results


def process_directory(input_dir, output_dir, file_pattern="*.jpg",
                      calc_gradient_x=True, calc_gradient_y=True,
                      calc_edge_strength=True, calc_edge_orientation=True,
                      ksize=3):
    """
    Process all images in a directory with Sobel operators.

    Parameters:
    -----------
    input_dir : str
        Directory containing input images
    output_dir : str
        Directory to save output images
    file_pattern : str
        Pattern to match input files
    calc_gradient_x, calc_gradient_y, calc_edge_strength,
        calc_edge_orientation : bool
        Options to control which outputs are calculated
    ksize : int
        Size of the Sobel kernel
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all matching files
    input_path = Path(input_dir)
    files = list(input_path.glob(file_pattern))

    if not files:
        print(f"No files matching {file_pattern} found in {input_dir}")
        return

    # Create Sobel processor
    processor = SobelProcessor(ksize=ksize)

    # Process each file
    for file_path in tqdm(files, desc="Processing images with Sobel"):
        try:
            # Read image
            image = cv2.imread(str(file_path))
            if image is None:
                print(f"Could not read {file_path}")
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Process with Sobel
            results = processor.process_image(
                gray,
                calc_gradient_x=calc_gradient_x,
                calc_gradient_y=calc_gradient_y,
                calc_edge_strength=calc_edge_strength,
                calc_edge_orientation=calc_edge_orientation
            )

            # Save results
            base_filename = file_path.stem
            save_sobel_results(output_dir, base_filename, results)

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    print(f"Processed {len(files)} images. Results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage for a single image
    image_path = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/images/14_2.jpg"
    output_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/sobel_features/"

    # Example 1: Calculate all Sobel features
    print("Example 1: All Sobel features")
    process_single_image(
        image_path=image_path,
        output_dir=output_dir,
        visualize=True,
        calc_gradient_x=True,
        calc_gradient_y=True,
        calc_edge_strength=True,
        calc_edge_orientation=True,
        ksize=3
    )

    # Example 2: Only edge strength and orientation
    # print("\nExample 2: Only edge strength and orientation")
    # process_single_image(
    #     image_path=image_path,
    #     output_dir=output_dir,
    #     visualize=True,
    #     calc_gradient_x=False,
    #     calc_gradient_y=False,
    #     calc_edge_strength=True,
    #     calc_edge_orientation=True,
    #     ksize=3
    # )

    # Example 3: Only gradients in X and Y direction
    # print("\nExample 3: Only gradients in X and Y direction")
    # process_single_image(
    #     image_path=image_path,
    #     output_dir=output_dir,
    #     visualize=True,
    #     calc_gradient_x=True,
    #     calc_gradient_y=True,
    #     calc_edge_strength=False,
    #     calc_edge_orientation=False,
    #     ksize=3
    # )

    # Example 4: Different kernel size
    # print("\nExample 4: Larger kernel size (ksize=5)")
    # process_single_image(
    #     image_path=image_path,
    #     output_dir=output_dir,
    #     visualize=True,
    #     calc_gradient_x=False,
    #     calc_gradient_y=False,
    #     calc_edge_strength=True,
    #     calc_edge_orientation=False,
    #     ksize=5
    # )

    # Example for batch processing - commented out to avoid accidental exec.
    """
    process_directory(
        input_dir="C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/images/",
        output_dir="C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
doctorado/CrackDataset/luz_crack/sobel_features/",
        file_pattern="*.jpg",
        calc_gradient_x=True,
        calc_gradient_y=True,
        calc_edge_strength=True,
        calc_edge_orientation=True,
        ksize=3
    )
    """
