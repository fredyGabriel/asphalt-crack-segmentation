import os
import cv2
from tqdm import tqdm
import numpy as np


def process_images_canny(input_folder, output_folder, lower_threshold=50,
                         upper_threshold=150, aperture_size=3,
                         apply_blur=True, kernel_size=5, overwrite=True):
    """
    Processes images applying edge detection with the Canny method.
    Optimized to detect cracks in asphalt pavement.

    Args:
        input_folder (str): Path to folder with original images
        output_folder (str): Path where processed images will be saved
        lower_threshold (int): Lower threshold for Canny hysteresis
        upper_threshold (int): Upper threshold for Canny hysteresis
        aperture_size (int): Aperture size for the Sobel operator (3, 5 or 7)
        apply_blur (bool): Whether to apply Gaussian blur first
        kernel_size (int): Kernel size for blur (must be odd)
        overwrite (bool): If True, overwrites existing files
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Valid image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # List all files in the input folder
    files = [f for f in os.listdir(input_folder)
             if os.path.splitext(f.lower())[1] in valid_extensions]

    if not files:
        print(f"No image files found in {input_folder}")
        return

    print(f"Processing {len(files)} images with Canny detector...")

    # Process each image with progress bar
    for filename in tqdm(files):
        input_path = os.path.join(input_folder, filename)

        # Change output extension to PNG
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(output_folder, output_filename)

        # Check if file already exists and if we shouldn't overwrite
        if os.path.exists(output_path) and not overwrite:
            print(f"Skipping {filename} (already exists)")
            continue

        try:
            # Read image with OpenCV
            img = cv2.imread(input_path)
            if img is None:
                print(f"Could not read image: {filename}")
                continue

            # Convert to grayscale if the image is in color
            if len(img.shape) > 2:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = img

            # Apply Gaussian blur to reduce noise (optional)
            if apply_blur:
                gray_img = cv2.GaussianBlur(gray_img,
                                            (kernel_size, kernel_size), 0)

            # Apply Canny edge detector
            edges = cv2.Canny(gray_img,
                              threshold1=lower_threshold,
                              threshold2=upper_threshold,
                              apertureSize=aperture_size,
                              L2gradient=True)

            # Save processed image in PNG format
            cv2.imwrite(output_path, edges)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")


def process_single_image_canny(input_image, output_path=None,
                               lower_threshold=50,
                               upper_threshold=150, aperture_size=3,
                               apply_blur=True, kernel_size=5):
    """
    Process a single image applying edge detection with the Canny method.
    Optimized to detect cracks in asphalt pavement.

    Args:
        input_image (str or numpy.ndarray): Path to image or image as numpy
            array
        output_path (str, optional): Path where processed image will be saved
                                    If None, the function only returns the
                                    result
        lower_threshold (int): Lower threshold for Canny hysteresis
        upper_threshold (int): Upper threshold for Canny hysteresis
        aperture_size (int): Aperture size for the Sobel operator (3, 5 or 7)
        apply_blur (bool): Whether to apply Gaussian blur first
        kernel_size (int): Kernel size for blur (must be odd)

    Returns:
        numpy.ndarray: Image with detected edges
    """
    # Read image if path is provided, otherwise use the array
    if isinstance(input_image, str):
        img = cv2.imread(input_image)
        if img is None:
            raise ValueError(f"Could not read image: {input_image}")
    else:
        img = input_image.copy()

    # Convert to grayscale if the image is in color
    if len(img.shape) > 2:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # Apply Gaussian blur to reduce noise (optional)
    if apply_blur:
        gray_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)

    # Apply Canny edge detector
    edges = cv2.Canny(gray_img,
                      threshold1=lower_threshold,
                      threshold2=upper_threshold,
                      apertureSize=aperture_size,
                      L2gradient=True)

    # Save the processed image if output path is provided
    if output_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)),
                    exist_ok=True)
        cv2.imwrite(output_path, edges)

    return edges


if __name__ == "__main__":
    # Example usage for multiple images
# flake8: noqa
#     input_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
# CrackDataset/luz_crack/4-clahe_enhanced/"
#     output_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
# CrackDataset/luz_crack/5-canny_edges/"

#     Process multiple images
#     process_images_canny(
#         input_dir,
#         output_dir,
#         lower_threshold=250,
#         upper_threshold=350,
#         aperture_size=3,
#         apply_blur=True,
#         kernel_size=5,
#         overwrite=True
#     )
#     print("Canny detector processing completed! Images saved in PNG format")
# end flake8: noqa

    # Example usage for a single image
    single_img_path = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
doctorado/CrackDataset/luz_crack/images/14_2.jpg"
    single_output_path = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
doctorado/CrackDataset/luz_crack/5-canny_edges/single_14_2.png"

    # Process single image
    edges = process_single_image_canny(
        single_img_path,
        single_output_path,
        lower_threshold=250,
        upper_threshold=350,
        aperture_size=3,
        apply_blur=True,
        kernel_size=5
    )
    print(f"Single image processed and saved to {single_output_path}")

    # Display the original image alongside the edge detection result
    original_img = cv2.imread(single_img_path)

    # Resize large images for better display
    max_height = 600
    if original_img.shape[0] > max_height:
        scale = max_height / original_img.shape[0]
        original_img = cv2.resize(original_img, None, fx=scale, fy=scale)
        edges = cv2.resize(edges, None, fx=scale, fy=scale)

    # Convert edges to 3-channel (BGR) for consistent display
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Create side-by-side comparison
    comparison = cv2.hconcat([original_img, edges_colored])

    # Add titles
    title_height = 30
    title_image = np.ones((title_height, comparison.shape[1], 3),
                          dtype=np.uint8) * 255

    # Add text to the title image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(title_image, "Original Image", (original_img.shape[1]//4, 20),
                font, 0.7, (0, 0, 0), 2)
    cv2.putText(title_image, "Canny Edge Detection", 
                (original_img.shape[1] + original_img.shape[1]//4, 20),
                font, 0.7, (0, 0, 0), 2)

    # Combine title and comparison images
    final_display = cv2.vconcat([title_image, comparison])

    # Show the comparison
    cv2.imshow('Canny Edge Detection Comparison', final_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
