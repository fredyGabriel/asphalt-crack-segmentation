import os
import cv2
from tqdm import tqdm


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


if __name__ == "__main__":
    input_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/4-clahe_enhanced/"
    output_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/5-canny_edges/"

    # Optimized parameters for crack detection in pavement:
    # - lower_threshold=50: Sensitive to weak edges (thin cracks)
    # - upper_threshold=150: Rejects noise but keeps important edges
    # - aperture_size=3: Good precision for fine details
    # - apply_blur=True: Reduces noise before detection
    # - kernel_size=5: Balanced size for removing noise without losing details

    process_images_canny(
        input_dir,
        output_dir,
        lower_threshold=250,
        upper_threshold=350,
        aperture_size=3,
        apply_blur=True,
        kernel_size=5,
        overwrite=True
    )
    print("Canny detector processing completed! Images saved in PNG format")
