import os
import cv2
import numpy as np
from tqdm import tqdm


def process_images_adaptive_threshold(input_folder, output_folder,
                                      block_size=11, c_value=2,
                                      apply_blur=True, blur_kernel_size=5,
                                      invert=True, morphology=True,
                                      morph_type='close', morph_kernel_size=3,
                                      overwrite=True):
    """
    Processes images applying Gaussian adaptive thresholding.
    This technique calculates local thresholds for each region of the image,
    using a Gaussian weighting for neighboring pixels.

    Args:
        input_folder (str): Path to folder with original images
        output_folder (str): Path where processed images will be saved
        block_size (int): Size of the neighborhood block (must be odd)
        c_value (int): Constant subtracted from the weighted average
        apply_blur (bool): Whether to apply Gaussian blur first
        blur_kernel_size (int): Kernel size for blur
        invert (bool): Whether to invert the resulting image (cracks in white)
        morphology (bool): Whether to apply morphological operations
        morph_type (str): Type of morphological operation ('open', 'close',
            'dilate', 'erode', 'open-close', 'close-open')
        morph_kernel_size (int): Kernel size for morphological operations
        overwrite (bool): If True, overwrites existing files
    """
    # Check that block_size is odd
    if block_size % 2 == 0:
        block_size += 1
        print(f"Adjusting block_size to {block_size} (must be odd)")

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

    print(f"Processing {len(files)} images with adaptive thresholding...")

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
                                            (blur_kernel_size,
                                             blur_kernel_size), 0)

            # Apply Gaussian adaptive thresholding
            # max_value=255: value assigned to pixels that exceed the threshold
            # adaptiveMethod=ADAPTIVE_THRESH_GAUSSIAN_C: uses Gaussian
            # weighting
            # thresholdType=THRESH_BINARY: simple binarization (>threshold ->
            # max_value, <=threshold -> 0)
            # blockSize: neighborhood size for calculating threshold
            # C: constant subtracted from the calculated average
            thresholded = cv2.adaptiveThreshold(
                gray_img,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                c_value
            )

            # Invert image if needed (cracks in white)
            if invert:
                thresholded = cv2.bitwise_not(thresholded)

            # Apply morphological operations to improve segmentation
            if morphology:
                kernel = np.ones((morph_kernel_size, morph_kernel_size),
                                 np.uint8)

                if morph_type == 'open':
                    # Removes small foreground objects
                    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN,
                                                   kernel)
                elif morph_type == 'close':
                    # Fills small holes and closes gaps in cracks
                    thresholded = cv2.morphologyEx(thresholded,
                                                   cv2.MORPH_CLOSE, kernel)
                elif morph_type == 'open-close':
                    # First removes noise, then joins nearby components
                    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN,
                                              kernel)
                    thresholded = cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                                                   kernel)
                elif morph_type == 'close-open':
                    # First joins components, then removes noise
                    closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE,
                                              kernel)
                    thresholded = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                                                   kernel)
                elif morph_type == 'dilate':
                    # Expands detected cracks
                    thresholded = cv2.dilate(thresholded, kernel, iterations=1)
                elif morph_type == 'erode':
                    # Thins detected cracks
                    thresholded = cv2.erode(thresholded, kernel, iterations=1)

            # Save processed image in PNG format
            cv2.imwrite(output_path, thresholded)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    input_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/4-clahe_enhanced/"
    output_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/5c-gauss_adaptive_threshold/"

    # Optimized parameters for asphalt pavement cracks:
    process_images_adaptive_threshold(
        input_dir,
        output_dir,
        block_size=25,            # Large block size to adapt to variations
        c_value=5,                # Moderate C value for balance
        apply_blur=True,          # Smooths image before thresholding
        blur_kernel_size=9,       # Removes noise while keeping details
        invert=True,              # Cracks in white for subsequent analysis
        morphology=False,         # Morphological operation to improve segment.
        morph_type='close-open',  # First joins cracks, then removes noise
        morph_kernel_size=3,      # Moderate size to preserve fine details
        overwrite=True
    )
    print("Gaussian adaptive thresholding processing completed! \
Images saved in PNG format")
