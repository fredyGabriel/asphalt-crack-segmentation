import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


def apply_clahe_to_image(input_image, output_path=None, clip_limit=2.0,
                         tile_grid_size=(8, 8), visualize=False,
                         convert_to_grayscale=False):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a single
    image.

    Parameters:
    -----------
    input_image : str or numpy.ndarray
        Path to image or image as numpy array
    output_path : str, optional
        Path where processed image will be saved
        If None, the function only returns the result
    clip_limit : float
        Threshold for contrast limiting
    tile_grid_size : tuple
        Size of grid for histogram equalization
    visualize : bool
        Whether to display the original and enhanced images
    convert_to_grayscale : bool
        If True, converts image to grayscale before applying CLAHE

    Returns:
    --------
    numpy.ndarray
        CLAHE enhanced image
    """
    # Read image if path is provided, otherwise use the array
    if isinstance(input_image, str):
        img = cv2.imread(input_image)
        if img is None:
            raise ValueError(f"Could not read image: {input_image}")
    else:
        img = input_image.copy()

    # Store original for visualization
    original_img = img.copy()

    # Check if we need to convert to grayscale
    if convert_to_grayscale and len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        is_color = False
    else:
        is_color = len(img.shape) > 2 and img.shape[2] == 3

    if is_color:
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=tile_grid_size)
        enhanced_l = clahe.apply(l)

        # Merge back the channels
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        # Convert back to BGR
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        # Apply CLAHE directly to grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=tile_grid_size)
        enhanced = clahe.apply(img)

    # Save the enhanced image if output path is provided
    if output_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)),
                    exist_ok=True)
        cv2.imwrite(output_path, enhanced)

    # Visualize if requested
    if visualize:
        # Use matplotlib for display
        plt.figure(figsize=(12, 6))

        if convert_to_grayscale:
            # Show grayscale version of original in the first panel
            if len(original_img.shape) > 2:
                orig_display = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            else:
                orig_display = original_img
            orig_is_color = False
        else:
            # Show original color image
            if len(original_img.shape) > 2 and original_img.shape[2] == 3:
                orig_display = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                orig_is_color = True
            else:
                orig_display = original_img
                orig_is_color = False

        enhanced_is_color = len(enhanced.shape) > 2 and enhanced.shape[2] == 3
        if enhanced_is_color:
            enhanced_display = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        else:
            enhanced_display = enhanced

        plt.subplot(1, 2, 1)
        plt.imshow(orig_display, cmap='gray' if not orig_is_color else None)
        plt.title('Original Image' + (' (Grayscale)' if convert_to_grayscale
                                      else ''))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(enhanced_display, cmap='gray' if not enhanced_is_color else
                   None)
        title = 'CLAHE Enhanced (Grayscale)' if convert_to_grayscale else\
            'CLAHE Enhanced'
        plt.title(title)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return enhanced


def process_images_clahe(input_folder, output_folder, clip_limit=2.0,
                         tile_grid_size=(8, 8), overwrite=True,
                         convert_to_grayscale=False):
    """
    Processes a folder of images applying CLAHE enhancement.

    Parameters:
    -----------
    input_folder : str
        Path to folder with original images
    output_folder : str
        Path where processed images will be saved
    clip_limit : float
        Threshold for contrast limiting
    tile_grid_size : tuple
        Size of grid for histogram equalization
    overwrite : bool
        If True, overwrites existing files
    convert_to_grayscale : bool
        If True, converts images to grayscale before applying CLAHE
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

    mode_str = "grayscale " if convert_to_grayscale else ""
    print(f"Processing {len(files)} images with {mode_str} CLAHE \
enhancement...")

    # Process each image with progress bar
    for filename in tqdm(files):
        input_path = os.path.join(input_folder, filename)

        # Keep the same extension if PNG, otherwise convert to PNG
        if os.path.splitext(filename.lower())[1] == '.png':
            output_filename = filename
        else:
            output_filename = os.path.splitext(filename)[0] + '.png'

        output_path = os.path.join(output_folder, output_filename)

        # Check if file already exists and if we shouldn't overwrite
        if os.path.exists(output_path) and not overwrite:
            print(f"Skipping {filename} (already exists)")
            continue

        try:
            # Apply CLAHE to the image
            apply_clahe_to_image(input_path, output_path, clip_limit,
                                 tile_grid_size,
                                 convert_to_grayscale=convert_to_grayscale)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    print(f"CLAHE enhancement completed! Enhanced images saved to \
{output_folder}")


if __name__ == "__main__":
    # Example usage for a single image
    image_path = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/images/14_2.jpg"
    output_path = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/clahe_enhanced/14_2_clahe.png"
    grayscale_output_path = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
doctorado/CrackDataset/luz_crack/clahe_enhanced/14_2_clahe_gray.png"

    # Process single image with CLAHE
    enhanced_gray = apply_clahe_to_image(
        image_path,
        grayscale_output_path,
        clip_limit=10.0,
        tile_grid_size=(8, 8),
        visualize=True,
        convert_to_grayscale=True  # Fase for color image
    )
    print(f"Enhanced grayscale image saved to {grayscale_output_path}")

    # Example for batch processing
    """
    process_images_clahe(
        input_folder="C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
doctorado/CrackDataset/luz_crack/images/",
        output_folder="C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
doctorado/CrackDataset/luz_crack/clahe_enhanced/",
        clip_limit=2.0,
        tile_grid_size=(8, 8),
        overwrite=True,
        convert_to_grayscale=False  # Set to True to process in grayscale
    )
    """
