import os
import cv2
from tqdm import tqdm


def process_images_clahe(input_folder, output_folder, clip_limit=2.0,
                         tile_grid_size=(8, 8), overwrite=True):
    """
    Processes images from a folder applying Contrast Limited Adaptive
        Histogram Equalization (CLAHE).
    CLAHE enhances local contrast and highlights details while preserving
        global information.

    Args:
        input_folder (str): Path to folder with original images
        output_folder (str): Path where processed images will be saved
        clip_limit (float): Contrast limit for clipping (>0)
        tile_grid_size (tuple): Grid size for adaptive processing
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

    print(f"Processing {len(files)} images with CLAHE...")

    # Create the CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

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

            # Apply CLAHE
            enhanced_img = clahe.apply(gray_img)

            # Save processed image in PNG format
            cv2.imwrite(output_path, enhanced_img)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    input_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/3-median_filtered/"
    output_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/4-clahe_enhanced/"

    # Recommended parameters for pavement cracks:
    # - clip_limit=3.0: Enhances contrast without exaggerating
    # - tile_grid_size=(8, 8): Grid size for local detail

    process_images_clahe(
        input_dir,
        output_dir,
        clip_limit=3.0,
        tile_grid_size=(8, 8),
        overwrite=True
    )
    print("CLAHE processing completed! Images saved in PNG format")
