import os
import cv2
from tqdm import tqdm


def process_images_sobel(input_folder, output_folder, ksize=3,
                         apply_blur=True, blur_kernel_size=5,
                         threshold=None, combine_method='magnitude',
                         scale_factor=1.0, overwrite=True):
    """
    Processes images applying edge detection with the Sobel operator.
    Optimized to detect cracks in asphalt pavement.

    Args:
        input_folder (str): Path to folder with original images
        output_folder (str): Path where processed images will be saved
        ksize (int): Size of the Sobel kernel (must be 1, 3, 5 or 7)
        apply_blur (bool): Whether to apply Gaussian blur first
        blur_kernel_size (int): Kernel size for blur
        threshold (int): Threshold for binarization (None for no binarization)
        combine_method (str): Method to combine gradients ('magnitude',
            'xy_sum', 'x_only', 'y_only')
        scale_factor (float): Factor to scale the intensity of the result
        overwrite (bool): If True, overwrites existing files
    """
    # Verify valid ksize (must be 1, 3, 5 or 7)
    valid_ksizes = [1, 3, 5, 7]
    if ksize not in valid_ksizes:
        ksize = min(valid_ksizes, key=lambda x: abs(x - ksize))
        print(f"Adjusting ksize to {ksize} (valid values: 1, 3, 5, 7)")

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

    print(f"Processing {len(files)} images with Sobel operator...")

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
            # Read the image with OpenCV
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

            # Calculate gradients with Sobel in X and Y directions
            grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=ksize)
            grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=ksize)

            # Combine gradients according to the chosen method
            if combine_method == 'magnitude':
                # Gradient magnitude (Pythagoras)
                sobel = cv2.magnitude(grad_x, grad_y)
            elif combine_method == 'xy_sum':
                # Sum of absolute values
                sobel = cv2.convertScaleAbs(grad_x) + cv2.convertScaleAbs(
                    grad_y)
            elif combine_method == 'x_only':
                # Only gradient in X (highlights vertical edges)
                sobel = cv2.convertScaleAbs(grad_x)
            elif combine_method == 'y_only':
                # Only gradient in Y (highlights horizontal edges)
                sobel = cv2.convertScaleAbs(grad_y)
            else:
                # By default, use magnitude
                sobel = cv2.magnitude(grad_x, grad_y)

            # Scale the result to improve visualization
            sobel = cv2.convertScaleAbs(sobel * scale_factor)

            # Apply thresholding if a threshold was specified
            if threshold is not None:
                _, sobel = cv2.threshold(sobel, threshold, 255,
                                         cv2.THRESH_BINARY)

            # Save processed image in PNG format
            cv2.imwrite(output_path, sobel)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    input_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/images/"
    output_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/5-sobel_edges/"

    # Optimized parameters for crack detection in pavement:
    process_images_sobel(
        input_dir,
        output_dir,
        ksize=3,                  # Sobel kernel size (3 offers good detail)
        apply_blur=True,          # Reduce noise before applying Sobel
        blur_kernel_size=7,       # Moderate blur kernel
        threshold=240,            # Threshold for binarization
        combine_method='magnitude',  # (magnitude is more precise)
        scale_factor=2.0,         # Scale factor to improve visualization
        overwrite=True
    )
    print("Sobel operator processing completed! Images saved in PNG format")
