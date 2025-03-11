import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import measure


def analyze_image(binary_img):
    """
    Analyzes a binary image and determines the best morphological operation.

    Args:
        binary_img: Binary image (cracks in white)

    Returns:
        tuple: (recommended operation, recommended kernel size)
    """
    # Analysis of connected components
    labeled_img = measure.label(binary_img)
    regions = measure.regionprops(labeled_img)

    # Extract key features
    total_pixels = binary_img.size
    white_pixels = np.count_nonzero(binary_img)
    crack_density = white_pixels / total_pixels

    # Number of components and size distribution
    num_components = len(regions)
    if num_components == 0:
        return 'dilate', 3  # No cracks detected, try dilation

    # Analyze component sizes
    areas = [region.area for region in regions]
    avg_area = np.mean(areas)
    small_components_ratio = np.sum([1 for a in areas if a < 20]) /\
        num_components

    # Analyze component shape
    eccentricities = [region.eccentricity for region in regions if
                      region.area > 10]
    avg_eccentricity = np.mean(eccentricities) if eccentricities else 0.5

    # Spacing between components
    # This is simplified, a real analysis could use distances between centroids
    spacing_metric = num_components / crack_density if crack_density > 0 else\
        100

    # Decision logic based on analyzed features

    # Case 1: Many small components (noise)
    if small_components_ratio > 0.7 and num_components > 50:
        return 'open', 3  # Opening to remove noise

    # Case 2: Large components with good eccentricity but disconnected
    if avg_eccentricity > 0.8 and spacing_metric < 5000 and crack_density >\
            0.001:
        return 'close', 3  # Closing to connect nearby cracks

    # Case 3: Very thick cracks
    if avg_area > 100 and crack_density > 0.05:
        return 'erode', 2  # Erosion to thin thick cracks

    # Case 4: Few cracks detected or very thin
    if crack_density < 0.01 or avg_area < 20:
        return 'dilate', 2  # Dilation to make cracks more visible

    # Default case: combine operations
    if avg_eccentricity > 0.6:
        return 'close-open', 3  # Closing then opening to clean and connect

    return 'close', 3  # Conservative operation by default


def process_images_adaptive_morphology(input_folder, output_folder,
                                       overwrite=True,
                                       visualize_decision=False):
    """
    Processes images applying adaptive morphological operations.

    Args:
        input_folder (str): Path to folder with binary images
        output_folder (str): Path where processed images will be saved
        overwrite (bool): If True, overwrites existing files
        visualize_decision (bool): If True, generates visualization of analysis
    """
    # Create output and visualization folders if necessary
    os.makedirs(output_folder, exist_ok=True)
    if visualize_decision:
        viz_folder = os.path.join(output_folder, "analysis")
        os.makedirs(viz_folder, exist_ok=True)

    # Valid image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # List all files in the input folder
    files = [f for f in os.listdir(input_folder)
             if os.path.splitext(f.lower())[1] in valid_extensions]

    if not files:
        print(f"No image files found in {input_folder}")
        return

    print(f"Processing {len(files)} images with adaptive morphological \
operations...")

    # For statistics
    operations_used = {}

    # Process each image with progress bar
    for filename in tqdm(files):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Check if file already exists and if we shouldn't overwrite
        if os.path.exists(output_path) and not overwrite:
            print(f"Skipping {filename} (already exists)")
            continue

        try:
            # Read the image with OpenCV
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Could not read image: {filename}")
                continue

            # Polarity normalization: ensure cracks are ALWAYS white
            white_pixels = np.count_nonzero(img > 127)
            black_pixels = np.count_nonzero(img <= 127)

            # If there are more white pixels than black pixels, invert
            # (assuming cracks = minority)
            if white_pixels > black_pixels:
                img = 255 - img
                print(f"Image {filename} inverted to normalize polarity")

            # Recreate the binary representation after normalization
            binary = img > 127

            # Analyze the image and decide which operation to apply
            operation, kernel_size = analyze_image(binary)

            # Update statistics
            operations_used[operation] = operations_used.get(operation, 0) + 1

            # Create kernel for the operation
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Apply the chosen morphological operation
            if operation == 'open':
                result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            elif operation == 'close':
                result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            elif operation == 'dilate':
                result = cv2.dilate(img, kernel, iterations=1)
            elif operation == 'erode':
                result = cv2.erode(img, kernel, iterations=1)
            elif operation == 'close-open':
                # Combination of operations
                temp = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                result = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)
            else:
                result = img  # No changes

            # Save processed image
            cv2.imwrite(output_path, result)

            # Optional visualization
            if visualize_decision:
                plt.figure(figsize=(15, 5))

                plt.subplot(131)
                plt.imshow(img, cmap='gray')
                plt.title('Original')
                plt.axis('off')

                plt.subplot(132)
                plt.imshow(result, cmap='gray')
                plt.title(f'Operation: {operation}, Kernel: {kernel_size}x\
{kernel_size}')
                plt.axis('off')

                # Visual analysis
                labeled = measure.label(binary)
                plt.subplot(133)
                plt.imshow(labeled, cmap='nipy_spectral')
                plt.title(f'Analysis: {len(np.unique(labeled))-1} components')
                plt.axis('off')

                plt.tight_layout()
                viz_path = os.path.join(
                    viz_folder, f"analysis_{os.path.splitext(filename)[0]}.png"
                    )
                plt.savefig(viz_path)
                plt.close()

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    # Show statistics
    print("\nStatistics of applied operations:")
    for op, count in operations_used.items():
        percentage = count / len(files) * 100
        print(f"- {op}: {count} images ({percentage:.1f}%)")


if __name__ == "__main__":
    input_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/5b-otsu_threshold/"
    output_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/6-adaptive_morphology_from_otsu/"

    process_images_adaptive_morphology(
        input_dir,
        output_dir,
        overwrite=True,
        visualize_decision=False  # Generate analysis visualizations
    )
    print("Processing with adaptive morphological operations completed!")
