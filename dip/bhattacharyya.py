import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


class BhattacharyyaAnalyzer:
    """
    Class for analyzing the separability of crack vs. non-crack pixels
    using Bhattacharyya distance on image intensity distributions.
    """
    def __init__(
        self,
        bins=256,
        intensity_range=(0, 255),
        epsilon=1e-10
    ):
        """
        Initialize the Bhattacharyya analyzer with configurable parameters.

        Parameters:
        -----------
        bins : int
            Number of bins for histogram calculation
        intensity_range : tuple
            Range of intensity values (min, max)
        epsilon : float
            Small constant to avoid division by zero or log(0)
        """
        self.bins = bins
        self.intensity_range = intensity_range
        self.epsilon = epsilon

    def separate_pixels(self, image, mask):
        """
        Separate pixels into crack and non-crack groups based on the mask.

        Parameters:
        -----------
        image : ndarray
            Input grayscale image
        mask : ndarray
            Binary mask where 1 represents crack pixels and 0 represents
            background

        Returns:
        --------
        tuple
            (crack_pixels, non_crack_pixels) as flattened arrays
        """
        # Ensure image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ensure mask is binary
        mask_binary = (mask > 0).astype(np.uint8)

        # Flatten arrays
        image_flat = image.flatten()
        mask_flat = mask_binary.flatten()

        # Extract pixels by group
        crack_pixels = image_flat[mask_flat == 1]
        non_crack_pixels = image_flat[mask_flat == 0]

        return crack_pixels, non_crack_pixels

    def compute_histograms(self, crack_pixels, non_crack_pixels):
        """
        Compute normalized histograms for crack and non-crack pixel intensities

        Parameters:
        -----------
        crack_pixels : ndarray
            Array of intensity values for crack pixels
        non_crack_pixels : ndarray
            Array of intensity values for non-crack pixels

        Returns:
        --------
        tuple
            (hist_crack, hist_non_crack) normalized histograms
        """
        # Check if there are pixels in both groups
        if len(crack_pixels) == 0 or len(non_crack_pixels) == 0:
            raise ValueError("One or both pixel groups are empty. Cannot \
compute histograms.")

        # Compute histograms with density=True for proper normalization
        # This ensures the area under the histogram equals 1.0
        hist_crack, _ = np.histogram(
            crack_pixels,
            bins=self.bins,
            range=self.intensity_range,
            density=True
        )

        hist_non_crack, _ = np.histogram(
            non_crack_pixels,
            bins=self.bins,
            range=self.intensity_range,
            density=True
        )

        # Check for zero-sum histograms (extremely rare case)
        # Add a small epsilon if needed to avoid division by zero
        if np.sum(hist_crack) == 0:
            hist_crack = hist_crack + self.epsilon
        if np.sum(hist_non_crack) == 0:
            hist_non_crack = hist_non_crack + self.epsilon

        return hist_crack, hist_non_crack

    def calculate_bhattacharyya_distance(self, hist1, hist2):
        """
        Calculate the Bhattacharyya distance between two normalized histograms.

        Parameters:
        -----------
        hist1, hist2 : ndarray
            Normalized histograms

        Returns:
        --------
        tuple
            (distance, coefficient) Bhattacharyya distance and coefficient
        """
        # Calculate Bhattacharyya coefficient
        bc = np.sum(np.sqrt(hist1 * hist2))

        # Calculate Bhattacharyya distance
        bd = -np.log(bc + self.epsilon)

        return bd, bc

    def analyze_image(self, image, mask):
        """
        Perform full Bhattacharyya analysis on an image and its mask.

        Parameters:
        -----------
        image : ndarray
            Input grayscale image
        mask : ndarray
            Binary mask where 1 represents crack pixels and 0 represents
            background

        Returns:
        --------
        dict
            Dictionary with analysis results including distance, coefficient,
            and histograms
        """
        # Separate pixels
        crack_pixels, non_crack_pixels = self.separate_pixels(image, mask)

        # Calculate histograms
        hist_crack, hist_non_crack = self.compute_histograms(
            crack_pixels, non_crack_pixels)

        # Calculate Bhattacharyya distance
        distance, coefficient = self.calculate_bhattacharyya_distance(
            hist_crack, hist_non_crack)

        # Return results
        return {
            'distance': distance,
            'coefficient': coefficient,
            'hist_crack': hist_crack,
            'hist_non_crack': hist_non_crack,
            'crack_pixels': crack_pixels,
            'non_crack_pixels': non_crack_pixels,
            'crack_mean': np.mean(crack_pixels),
            'non_crack_mean': np.mean(non_crack_pixels),
            'crack_std': np.std(crack_pixels),
            'non_crack_std': np.std(non_crack_pixels),
            'crack_count': len(crack_pixels),
            'non_crack_count': len(non_crack_pixels)
        }


def visualize_bhattacharyya_results(image, mask, results, figure_size=(18, 10),
                                    preprocessed_image=None, method_name=None):
    """
    Visualize the results of Bhattacharyya analysis.

    Parameters:
    -----------
    image : ndarray
        Input grayscale image (original image)
    mask : ndarray
        Binary mask
    results : dict
        Dictionary with analysis results
    figure_size : tuple
        Size of the figure
    preprocessed_image : ndarray or None
        Preprocessed version of the image (if None, use the original image)
    method_name : str or None
        Name of the preprocessing method to show in the title
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=figure_size)

    # Set the main title if method name is provided
    if method_name:
        fig.suptitle(f"Bhattacharyya Analysis - {method_name}", fontsize=16,
                     y=0.98)

    # Plot original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Plot mask
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask')
    axes[0, 1].axis('off')

    # If preprocessed image is provided, show it instead of the overlay
    if preprocessed_image is not None:
        # Plot preprocessed image
        axes[0, 2].imshow(preprocessed_image, cmap='gray')
        axes[0, 2].set_title(f'Preprocessed Image{" (" + method_name + ")" if
                                                  method_name else ""}')
        axes[0, 2].axis('off')
    else:
        # Create crack overlay on original image
        masked_image = image.copy()
        if len(masked_image.shape) == 2:
            # Convert to 3-channel for colored overlay
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)

        # Create colored overlay (green for cracks)
        overlay = np.zeros_like(masked_image)
        if len(mask.shape) == 2:
            overlay[:, :, 1] = mask * 255  # Green channel

        # Blend images
        alpha = 0.5
        overlay_image = cv2.addWeighted(masked_image, 1, overlay, alpha, 0)
        axes[0, 2].imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Crack Overlay')
        axes[0, 2].axis('off')

    # Plot histograms
    bin_edges = np.linspace(0, 255, len(results['hist_crack']) + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    axes[1, 0].bar(bin_centers, results['hist_crack'], width=2, alpha=0.7,
                   color='green', label='Crack')
    axes[1, 0].bar(bin_centers, results['hist_non_crack'], width=2, alpha=0.7,
                   color='red', label='Non-Crack')
    axes[1, 0].set_title('Intensity Distributions')
    axes[1, 0].set_xlabel('Intensity')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].legend()

    # Plot distribution as line graphs for better comparison
    axes[1, 1].plot(bin_centers, results['hist_crack'], color='green',
                    label='Crack')
    axes[1, 1].plot(bin_centers, results['hist_non_crack'], color='red',
                    label='Non-Crack')
    axes[1, 1].fill_between(
        bin_centers, results['hist_crack'], results['hist_non_crack'],
        where=(results['hist_crack'] < results['hist_non_crack']),
        facecolor='gray', alpha=0.3)
    axes[1, 1].set_title('Overlapping Distributions')
    axes[1, 1].set_xlabel('Intensity')
    axes[1, 1].set_ylabel('Probability')
    axes[1, 1].legend()

    # Plot statistics and results
    axes[1, 2].axis('off')
    stats_text = (
        f"Bhattacharyya Distance: {results['distance']:.4f}\n"
        f"Bhattacharyya Coefficient: {results['coefficient']:.4f}\n\n"
        f"Crack Pixels: {results['crack_count']} \
({results['crack_count']/(results['crack_count']+results['non_crack_count']
                          )*100:.2f}%)\n"
        f"Non-Crack Pixels: {results['non_crack_count']} \
({results['non_crack_count']/(results['crack_count']+results['non_crack_count']
                              )*100:.2f}%)\n\n"
        f"Crack Mean Intensity: {results['crack_mean']:.2f}\n"
        f"Non-Crack Mean Intensity: {results['non_crack_mean']:.2f}\n\n"
        f"Crack Std Deviation: {results['crack_std']:.2f}\n"
        f"Non-Crack Std Deviation: {results['non_crack_std']:.2f}"
    )
    axes[1, 2].text(0, 0.5, stats_text, fontsize=12, va='center')

    # Add interpretation guide
    interpretation = ""
    if results['distance'] < 0.2:
        interpretation = "Poor Separability: Distributions highly overlap"
    elif results['distance'] < 0.5:
        interpretation = "Moderate Separability: Some distinct features"
    elif results['distance'] < 0.8:
        interpretation = "Good Separability: Distributions fairly distinct"
    else:
        interpretation = "Excellent Separability: Distributions well separated"

    axes[1, 2].text(0, 0.1, f"Interpretation: {interpretation}", fontsize=12,
                    color='blue', weight='bold')

    plt.tight_layout()
    if method_name:
        plt.subplots_adjust(top=0.93)  # Make room for the suptitle
    plt.show()

    return fig


def process_single_image(image_path, mask_path, output_dir=None,
                         visualize=True, bins=256):
    """
    Process a single image and its mask to calculate Bhattacharyya distance.

    Parameters:
    -----------
    image_path : str
        Path to the grayscale image
    mask_path : str
        Path to the binary mask
    output_dir : str or None
        Directory to save outputs (None for no saving)
    visualize : bool
        Whether to display visualizations
    bins : int
        Number of bins for histogram calculation

    Returns:
    --------
    dict
        Analysis results
    """
    # Load image and mask
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    # Create analyzer
    analyzer = BhattacharyyaAnalyzer(bins=bins)

    # Analyze image
    results = analyzer.analyze_image(image, mask)

    # Visualize if requested
    if visualize:
        fig = visualize_bhattacharyya_results(image, mask, results)

        # Save figure if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            fig_path = os.path.join(output_dir, f"{base_filename}\
_bhattacharyya_analysis.png")
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved analysis to {fig_path}")

    return results


def process_directory(image_dir, mask_dir, output_dir, file_pattern="*.jpg",
                      bins=256, visualize_individual=False,
                      calculate_preprocessing_stats=False):
    """
    Process all matching images in a directory.

    Parameters:
    -----------
    image_dir : str
        Directory containing input images
    mask_dir : str
        Directory containing corresponding masks
    output_dir : str
        Directory to save outputs
    file_pattern : str
        Pattern to match input files
    bins : int
        Number of bins for histogram calculation
    visualize_individual : bool
        Whether to generate and save visualizations for each individual image
    calculate_preprocessing_stats : bool
        Whether to calculate statistics for different preprocessing methods

    Returns:
    --------
    tuple
        (results_dict, summary_stats, preprocessing_stats) - Analysis results
        and statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all matching image files
    image_path = Path(image_dir)
    image_files = list(image_path.glob(file_pattern))

    if not image_files:
        print(f"No files matching {file_pattern} found in {image_dir}")
        return {}, {}, {}

    # Create analyzer
    analyzer = BhattacharyyaAnalyzer(bins=bins)

    # Process each file
    results_dict = {}

    # Lists to store metrics for calculation of statistics
    distances = []
    coefficients = []
    crack_means = []
    non_crack_means = []
    crack_stds = []
    non_crack_stds = []
    crack_percentages = []

    # Storage for preprocessing statistics
    preprocessing_stats = {}
    if calculate_preprocessing_stats:
        # Initialize storage for each preprocessing method
        preprocessing_stats = {
            'original': {'distances': []},
            'clahe': {'distances': []},
            'median_blur': {'distances': []},
            'gaussian_blur': {'distances': []}
        }

    # Create or overwrite CSV file with header
    with open(os.path.join(output_dir, 'bhattacharyya_results.csv'), 'w') as f:
        header = "filename,distance,coefficient,crack_mean,non_crack_mean,"
        header += "crack_std,non_crack_std,crack_count,non_crack_count,"
        header += "crack_percentage"

        if calculate_preprocessing_stats:
            header += ",clahe_distance,median_blur_distance, \
gaussian_blur_distance"

        f.write(header + "\n")

    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Construct mask path (assumes same filename but potentially
            # different extension)
            mask_file = Path(mask_dir) / f"{image_file.stem}.png"
            if not mask_file.exists():
                # Try other common extensions
                for ext in ['.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                    mask_file = Path(mask_dir) / f"{image_file.stem}{ext}"
                    if mask_file.exists():
                        break
                else:
                    print(f"No matching mask found for {image_file}")
                    continue

            # Load image and mask
            image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"Could not read image or mask for {image_file}")
                continue

            # Analyze original image
            results = analyzer.analyze_image(image, mask)
            results_dict[image_file.name] = results

            # Calculate crack percentage
            total_pixels = results['crack_count'] + results['non_crack_count']
            crack_percentage = (results['crack_count'] / total_pixels) * 100

            # Append values for statistics
            distances.append(results['distance'])
            coefficients.append(results['coefficient'])
            crack_means.append(results['crack_mean'])
            non_crack_means.append(results['non_crack_mean'])
            crack_stds.append(results['crack_std'])
            non_crack_stds.append(results['non_crack_std'])
            crack_percentages.append(crack_percentage)

            # Store original image distance for preprocessing comparison
            if calculate_preprocessing_stats:
                preprocessing_stats['original']['distances'].\
                    append(results['distance'])

                # Calculate distance for each preprocessing method
                # CLAHE
                clahe_img = cv2.createCLAHE(
                    clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
                clahe_results = analyzer.analyze_image(clahe_img, mask)
                preprocessing_stats['clahe']['distances'].append(
                    clahe_results['distance'])

                # Median Blur
                median_blur_img = cv2.medianBlur(image, 5)
                median_blur_results = analyzer.analyze_image(median_blur_img,
                                                             mask)
                preprocessing_stats['median_blur']['distances'].append(
                    median_blur_results['distance'])

                # Gaussian Blur
                gaussian_blur_img = cv2.GaussianBlur(image, (5, 5), 0)
                gaussian_blur_results = analyzer.analyze_image(
                    gaussian_blur_img, mask)
                preprocessing_stats['gaussian_blur']['distances'].append(
                    gaussian_blur_results['distance'])

            # Create visualization and save it only if requested
            if visualize_individual:
                fig = visualize_bhattacharyya_results(image, mask, results)
                fig_path = os.path.join(output_dir, f"{image_file.stem}\
_bhattacharyya_analysis.png")
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

            # Save numeric results to CSV
            with open(os.path.join(output_dir, 'bhattacharyya_results.csv'),
                      'a') as f:
                row = f"{image_file.name},{results['distance']},"
                row += f"{results['coefficient']},{results['crack_mean']},"
                row += f"{results['non_crack_mean']},{results['crack_std']},"
                row += f"{results['non_crack_std']},{results['crack_count']},"
                row += f"{results['non_crack_count']},{crack_percentage:.2f}"

                if calculate_preprocessing_stats:
                    row += f",{clahe_results['distance']}"
                    row += f",{median_blur_results['distance']}"
                    row += f",{gaussian_blur_results['distance']}"

                f.write(row + "\n")

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

    # Calculate summary statistics
    summary_stats = {}
    if distances:  # Only calculate statistics if we have data
        summary_stats = {
            'distance_mean': np.mean(distances),
            'distance_std': np.std(distances),
            'coefficient_mean': np.mean(coefficients),
            'coefficient_std': np.std(coefficients),
            'crack_mean_avg': np.mean(crack_means),
            'crack_mean_std': np.std(crack_means),
            'non_crack_mean_avg': np.mean(non_crack_means),
            'non_crack_mean_std': np.std(non_crack_means),
            'crack_std_avg': np.mean(crack_stds),
            'crack_std_std': np.std(crack_stds),
            'non_crack_std_avg': np.mean(non_crack_stds),
            'non_crack_std_std': np.std(non_crack_stds),
            'crack_percentage_avg': np.mean(crack_percentages),
            'crack_percentage_std': np.std(crack_percentages),
            'sample_count': len(distances)
        }

        # Display summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Number of images processed: {summary_stats['sample_count']}")
        print(f"Bhattacharyya Distance: {summary_stats['distance_mean']:.4f} \
± {summary_stats['distance_std']:.4f}")
        print(f"Bhattacharyya Coefficient: \
{summary_stats['coefficient_mean']:.4f} ± \
{summary_stats['coefficient_std']:.4f}")
        print(f"Crack Mean Intensity: {summary_stats['crack_mean_avg']:.2f} ± \
{summary_stats['crack_mean_std']:.2f}")
        print(f"Non-Crack Mean Intensity: \
{summary_stats['non_crack_mean_avg']:.2f} ± \
{summary_stats['non_crack_mean_std']:.2f}")
        print(f"Average Crack Percentage: \
{summary_stats['crack_percentage_avg']:.2f}% ± \
{summary_stats['crack_percentage_std']:.2f}%")
        print("="*50)

        # Save summary statistics to CSV
        with open(os.path.join(output_dir, 'bhattacharyya_summary_stats.csv'),
                  'w') as f:
            f.write("metric,mean,std_dev\n")
            for key in summary_stats:
                if key != 'sample_count':
                    f.write(f"{key},{summary_stats[key]},\
{summary_stats.get(key + '_std', 'N/A')}\n")
            f.write(f"sample_count,{summary_stats['sample_count']},N/A\n")

        # Create histogram of distances
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=20, color='skyblue', edgecolor='black')
        plt.axvline(summary_stats['distance_mean'], color='red',
                    linestyle='--',
                    label=f'Mean: {summary_stats["distance_mean"]:.4f}')
        plt.xlabel('Bhattacharyya Distance')
        plt.ylabel('Frequency')
        plt.title('Distribution of Bhattacharyya Distances Across Images')
        plt.legend()
        plt.grid(alpha=0.3)

        # Save histogram
        hist_path = os.path.join(output_dir,
                                 'bhattacharyya_distance_histogram.png')
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Calculate preprocessing statistics
        if calculate_preprocessing_stats:
            for method, data in preprocessing_stats.items():
                if data['distances']:
                    mean_dist = np.mean(data['distances'])
                    std_dist = np.std(data['distances'])
                    preprocessing_stats[method]['mean'] = mean_dist
                    preprocessing_stats[method]['std'] = std_dist

            # Print preprocessing statistics
            print("\n" + "="*50)
            print("PREPROCESSING METHOD STATISTICS")
            print("="*50)
            print(f"Original: {preprocessing_stats['original']['mean']:.4f} ± \
{preprocessing_stats['original']['std']:.4f}")
            print(f"CLAHE: {preprocessing_stats['clahe']['mean']:.4f} ± \
{preprocessing_stats['clahe']['std']:.4f}")
            print(f"Median Blur: \
{preprocessing_stats['median_blur']['mean']:.4f} ± \
{preprocessing_stats['median_blur']['std']:.4f}")
            print(f"Gaussian Blur: \
{preprocessing_stats['gaussian_blur']['mean']:.4f} ± \
{preprocessing_stats['gaussian_blur']['std']:.4f}")
            print("="*50)

            # Save preprocessing statistics to CSV
            with open(os.path.join(output_dir, 'preprocessing_stats.csv'),
                      'w') as f:
                f.write("method,mean,std_dev\n")
                for method, data in preprocessing_stats.items():
                    if 'mean' in data:
                        f.write(f"{method},{data['mean']},{data['std']}\n")

            # Create comparison bar chart
            plt.figure(figsize=(10, 6))
            methods = list(preprocessing_stats.keys())
            means = [preprocessing_stats[m]['mean'] for m in methods]
            stds = [preprocessing_stats[m]['std'] for m in methods]

            bars = plt.bar(methods, means, yerr=stds, capsize=10)
            plt.ylabel('Bhattacharyya Distance')
            plt.title('Bhattacharyya Distance by Preprocessing Method')
            plt.grid(axis='y', alpha=0.3)

            # Add values to bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                         f'{mean:.4f}±{std:.4f}', ha='center', va='bottom',
                         fontsize=9, rotation=0)

            plt.tight_layout()
            comparison_path = os.path.join(output_dir,
                                           'preprocessing_comparison.png')
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()

    return results_dict, summary_stats, preprocessing_stats


def compare_preprocessing_methods(image_path, mask_path, preprocessors=None,
                                  output_dir=None, bins=256):
    """
    Compare different preprocessing methods using Bhattacharyya distance.

    Parameters:
    -----------
    image_path : str
        Path to the original image
    mask_path : str
        Path to the binary mask
    preprocessors : list
        List of preprocessing functions to apply
    output_dir : str or None
        Directory to save outputs (None for no saving)
    bins : int
        Number of bins for histogram calculation

    Returns:
    --------
    dict
        Dictionary mapping preprocessor names to analysis results
    """
    # Default preprocessors if none provided
    if preprocessors is None:
        preprocessors = {
            'Original': lambda img: img,
            'CLAHE': lambda img: cv2.createCLAHE(
                clipLimit=2.0, tileGridSize=(8, 8)).apply(img),
            'Gaussian Blur': lambda img: cv2.GaussianBlur(img, (5, 5), 0),
            'Median Blur': lambda img: cv2.medianBlur(img, 5),
            'Sobel Edge': lambda img: cv2.convertScaleAbs(
                cv2.Sobel(img, cv2.CV_16S, 1, 1, ksize=3))
        }

    # Load original image and mask
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    # Create analyzer
    analyzer = BhattacharyyaAnalyzer(bins=bins)

    # Process image with each preprocessor
    results_dict = {}
    distances = []
    preprocessor_names = []

    for name, preprocess_func in preprocessors.items():
        try:
            # Apply preprocessing
            processed_image = preprocess_func(image)

            # Analyze image
            results = analyzer.analyze_image(processed_image, mask)
            results_dict[name] = results
            distances.append(results['distance'])
            preprocessor_names.append(name)

            # Save individual result
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_filename = os.path.splitext(
                    os.path.basename(image_path))[0]

                # Save processed image
                img_path = os.path.join(output_dir, f"{base_filename}_\
{name.lower().replace(' ', '_')}.png")
                cv2.imwrite(img_path, processed_image)

                # Generate visualization - pass both original and processed
                # image
                fig = visualize_bhattacharyya_results(
                    image,  # Original image
                    mask,
                    results,
                    preprocessed_image=processed_image,  # Preprocessed image
                    method_name=name  # Method name for title
                )
                fig_path = os.path.join(output_dir, f"{base_filename}_\
{name.lower().replace(' ', '_')}_analysis.png")
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

        except Exception as e:
            print(f"Error processing with {name}: {str(e)}")

    # Create comparison chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(preprocessor_names, distances, color='skyblue')
    plt.xlabel('Bhattacharyya Distance')
    plt.ylabel('Preprocessing Method')
    plt.title('Bhattacharyya Distance Comparison of Preprocessing Methods')
    plt.grid(axis='x', alpha=0.3)

    # Add values to bars
    for bar, distance in zip(bars, distances):
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{distance:.4f}',
                 va='center', ha='left')

    plt.tight_layout()

    if output_dir:
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        comparison_path = os.path.join(output_dir, f"{base_filename}\
_method_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {comparison_path}")

    plt.show()

    return results_dict


if __name__ == "__main__":
    # Example usage

    image_path = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/images/12_1.jpg"
    mask_path = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/masks/12_1.png"
    output_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/bhattacharyya_results/"

    # Example 1: Process a single image
    print("Example 1: Processing single image")
    results = process_single_image(image_path, mask_path, output_dir)
    print(f"Bhattacharyya Distance: {results['distance']:.4f}")
    print(f"Bhattacharyya Coefficient: {results['coefficient']:.4f}")

    # Example 2: Compare different preprocessing methods
    print("\nExample 2: Comparing preprocessing methods")
    compare_results = compare_preprocessing_methods(
        image_path,
        mask_path,
        preprocessors={
            'Original': lambda img: img,
            'CLAHE': lambda img: cv2.createCLAHE(clipLimit=1.0,
                                                 tileGridSize=(8, 8)
                                                 ).apply(img),
            'Gaussian Blur': lambda img: cv2.GaussianBlur(img, (5, 5), 0),
            'Median Blur': lambda img: cv2.medianBlur(img, 5),
            'Sobel Edge': lambda img: cv2.convertScaleAbs(
                cv2.Sobel(img, cv2.CV_16S, 1, 1, ksize=3)),
            'Scharr Edge': lambda img: cv2.convertScaleAbs(
                cv2.Scharr(img, cv2.CV_16S, 1, 0) + cv2.Scharr(
                    img, cv2.CV_16S, 0, 1)),
            'Gabor Filter': lambda img: cv2.filter2D(
                img, cv2.CV_8UC3, cv2.getGaborKernel((21, 21), 4.0, np.pi/4,
                                                     10.0, 0.5, 0,
                                                     ktype=cv2.CV_32F))
        },
        output_dir=output_dir
    )

    # Example for batch processing - commented out to avoid accidental
    # execution
    """
    results, summary_stats, preprocessing_stats = process_directory(
        image_dir="C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/images/",
        mask_dir="C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/masks/",
        output_dir="C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
doctorado/CrackDataset/luz_crack/bhattacharyya_results/",
        file_pattern="*.jpg",
        visualize_individual=False,  # Set to False to skip individual visual.
        calculate_preprocessing_stats=True  # Calculate preprocessing stats.
    )
    """
