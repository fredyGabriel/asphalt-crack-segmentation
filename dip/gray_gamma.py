import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def process_images(input_folder, output_folder, gamma=1.2, overwrite=True):
    """
    Processes images from a folder, converting them to grayscale
    and applying gamma correction.

    Args:
        input_folder (str): Path to folder with original images
        output_folder (str): Path where processed images will be saved
        gamma (float): Gamma correction value (>1 darkens, <1 brightens)
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

    print(f"Processing {len(files)} images...")

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
            # Open image
            with Image.open(input_path) as img:
                # Convert to grayscale
                gray_img = img.convert('L')

                # Apply gamma correction
                img_array = np.array(gray_img, dtype=np.float32)
                img_array = 255.0 * (img_array / 255.0) ** (1.0 / gamma)
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)

                # Create new image from array
                processed_img = Image.fromarray(img_array)

                # Save processed image in PNG format
                processed_img.save(output_path, format='PNG')

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":

    input_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/images/"
    output_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/gray_gamma/"
    gamma = 0.9
    overwrite = True

    process_images(input_dir, output_dir, gamma, overwrite)
    print("Processing completed! Images saved in PNG format")
