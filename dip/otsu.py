import os
import cv2
import numpy as np
from tqdm import tqdm


def process_images_otsu(input_folder, output_folder, apply_blur=True,
                        kernel_size=5, invert=True, morphology=True,
                        morph_type='close', morph_kernel_size=5,
                        overwrite=True):
    """
    Processes images applying Otsu's optimal thresholding.
    This technique automatically finds the optimal threshold to binarize
    images.

    Args:
        input_folder (str): Path to folder with original images
        output_folder (str): Path where processed images will be saved
        apply_blur (bool): Whether to apply Gaussian blur first
        kernel_size (int): Kernel size for blur (must be odd)
        invert (bool): Whether to invert the resulting image (cracks in white)
        morphology (bool): Whether to apply morphological operations
        morph_type (str): Type of morphological operation ('open', 'close',
            'dilate', 'erode')
        morph_kernel_size (int): Kernel size for morphological operations
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

    print(f"Processing {len(files)} images with Otsu thresholding...")

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

            # Apply Otsu thresholding
            _, thresholded = cv2.threshold(gray_img, 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
                    # Joins small breaks and fills holes in cracks
                    thresholded = cv2.morphologyEx(thresholded,
                                                   cv2.MORPH_CLOSE, kernel)
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
CrackDataset/luz_crack/5b-otsu_threshold/"

    # Optimized parameters for asphalt pavement cracks:
    # - apply_blur=True: Smooths the image before thresholding
    # - kernel_size=5: Removes noise while keeping important details
    # - invert=True: Cracks in white (convenient for subsequent analysis)
    # - morphology=True: Applies morphological operation to improve
    # segmentation
    # - morph_type='close': Joins segments of discontinuous cracks
    # - morph_kernel_size=3: Moderate size to preserve fine details

    process_images_otsu(
        input_dir,
        output_dir,
        apply_blur=True,
        kernel_size=9,
        invert=True,
        morphology=True,
        morph_type='open-close',
        morph_kernel_size=3,
        overwrite=True
    )
    print("Otsu thresholding processing completed! Images saved in PNG format")
