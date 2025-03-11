import os
import cv2
from tqdm import tqdm


def process_images_bilateral(input_folder, output_folder, d=9, sigma_color=75,
                             sigma_space=75, overwrite=True):
    """
    Processes images from a folder by applying a bilateral filter.
    The bilateral filter preserves edges while reducing noise.

    Args:
        input_folder (str): Path to folder with original images
        output_folder (str): Path where processed images will be saved
        d (int): Diameter of each pixel neighborhood used during filtering
        sigma_color (float): Sigma in the color space. Larger values mix colors
            more
        sigma_space (float): Sigma in the coordinate space. Larger values mix
            farther pixels
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

    print(f"Processing {len(files)} images with bilateral filter...")

    # Process each image with progress bar
    for filename in tqdm(files):
        input_path = os.path.join(input_folder, filename)

        # Change output filename to PNG format
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(output_folder, output_filename)

        # Check if file already exists and if we shouldn't overwrite
        if os.path.exists(output_path) and not overwrite:
            print(f"Skipping {filename} (already exists)")
            continue

        try:
            # Read image with OpenCV (which reads it in BGR)
            img = cv2.imread(input_path)
            if img is None:
                print(f"Could not read image: {filename}")
                continue

            # Apply bilateral filter
            # d = pixel diameter, sigma_color = color space,
            # sigma_space = coordinate space
            filtered_img = cv2.bilateralFilter(img, d, sigma_color,
                                               sigma_space)

            # Save processed image in PNG format
            cv2.imwrite(output_path, filtered_img)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    input_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/gray_gamma/"
    output_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/bilateral_filtered/"

    # Optimal parameters for pavement cracks:
    # - d=9: balance between performance and quality
    # - sigma_color=75: preserves important color differences (cracks)
    # - sigma_space=75: allows influence from nearby pixels to smooth noise

    process_images_bilateral(
        input_dir,
        output_dir,
        d=9,
        sigma_color=75,
        sigma_space=75,
        overwrite=True
    )
    print("Bilateral filter processing completed! Images saved in PNG format")
