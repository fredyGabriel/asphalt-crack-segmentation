import os
import cv2
from tqdm import tqdm


def process_images_median(input_folder, output_folder, kernel_size=5,
                          overwrite=True):
    """
    Processes images from a folder applying a median filter.
    The median filter is particularly effective for removing "salt and pepper"
        noise while preserving edges.

    Args:
        input_folder (str): Path to folder with original images
        output_folder (str): Path where processed images will be saved
        kernel_size (int): Kernel size for the filter (must be an odd number)
        overwrite (bool): If True, overwrites existing files
    """
    # Check that kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
        print(f"Adjusting kernel size to {kernel_size} (must be odd)")

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

    print(f"Processing {len(files)} images with median filter...")

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

            # Apply median filter
            # The median filter replaces each pixel with the median of the
            # neighboring pixels
            filtered_img = cv2.medianBlur(img, kernel_size)

            # Save processed image in PNG format
            cv2.imwrite(output_path, filtered_img)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    input_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/2-bilateral_filtered/"
    output_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/3-median_filtered/"

    # Recommended parameters for pavement cracks:
    # - kernel_size=5: offers good balance between noise reduction and
    #   preservation of crack details

    process_images_median(
        input_dir,
        output_dir,
        kernel_size=5,
        overwrite=True
    )
    print("Median filter processing completed! Images saved in PNG format")
