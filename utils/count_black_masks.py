import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def count_black_masks(masks_folder):
    """
    Counts how many mask images in a folder are completely black.

    Args:
        masks_folder (str): Path to the folder with binary masks

    Returns:
        tuple: (number of black masks, total masks, list of black mask names)
    """
    # Valid image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # List all image files in the folder
    files = [f for f in os.listdir(masks_folder)
             if os.path.splitext(f.lower())[1] in valid_extensions]

    total_masks = len(files)
    black_masks = 0
    black_mask_names = []

    # Process each image with a progress bar
    for file in tqdm(files, desc="Analyzing masks"):
        full_path = os.path.join(masks_folder, file)

        try:
            # Load image
            # Ensure grayscale
            mask = Image.open(full_path).convert('L')

            # Convert to array and check if all pixels are 0
            mask_array = np.array(mask)
            if np.all(mask_array == 0):
                black_masks += 1
                black_mask_names.append(file)

        except Exception as e:
            print(f"Error processing {file}: {e}")

    return black_masks, total_masks, black_mask_names


if __name__ == "__main__":
    # Folder containing binary masks
    masks_folder = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
        doctorado/CrackDataset/luz_crack/masks/"

    # Count black masks
    num_black, total, black_list = count_black_masks(masks_folder)

    # Show results
    print("\nAnalysis results:")
    print(f"- Total masks analyzed: {total}")
    print(f"- Completely black masks: {num_black} \
({(num_black/total)*100:.2f}%)")

    # Optional: save the list of black masks to a file
    if num_black > 0:
        print("\nFirst 10 black masks found:")
        for name in black_list[:10]:
            print(f"- {name}")

        # Save complete list to a file
        file_path = os.path.join(os.path.dirname(masks_folder),
                                 "black_masks.txt")
        with open(file_path, 'w') as f:
            for name in black_list:
                f.write(f"{name}\n")
        print(f"\nComplete list saved to: {file_path}")
