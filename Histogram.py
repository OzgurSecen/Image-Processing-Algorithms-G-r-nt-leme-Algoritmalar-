import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_histogram_equalization(image):
    """Applies histogram equalization to the image."""
    return cv2.equalizeHist(image)

def apply_histogram_matching(source_image, reference_image):
    """Matches the histogram of the source image to that of the reference image."""
    source_hist, bins = np.histogram(source_image.flatten(), 256, [0, 256])
    reference_hist, _ = np.histogram(reference_image.flatten(), 256, [0, 256])

    source_cdf = source_hist.cumsum()
    source_cdf_normalized = source_cdf / source_cdf[-1]

    reference_cdf = reference_hist.cumsum()
    reference_cdf_normalized = reference_cdf / reference_cdf[-1]

    lookup_table = np.interp(source_cdf_normalized, reference_cdf_normalized, np.arange(256))

    matched_image = cv2.LUT(source_image, lookup_table.astype(np.uint8))
    return matched_image

def apply_contrast_stretching(image):
    """Applies contrast stretching to the image."""
    min_val, max_val = np.min(image), np.max(image)
    stretched_image = (image - min_val) * (255 / (max_val - min_val))
    return np.uint8(stretched_image)

def display_image(title, image):
    """Displays an image using Matplotlib."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()

def main():
    # Input Image Paths
    source_image_path = "resim.jpg"  # Path to your source image
    reference_image_path = "reference.jpg"  # Update with the correct path to your reference image

    try:
        # Load the source image
        source_image = cv2.imread(source_image_path, cv2.IMREAD_GRAYSCALE)
        if source_image is None:
            raise ValueError("Source image not found. Please check the path.")

        # Load the reference image
        reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
        if reference_image is None:
            print("Reference image not found. Skipping histogram matching.")
            reference_image = None

        # Apply and display histogram equalization
        equalized_image = apply_histogram_equalization(source_image)
        display_image("Histogram Equalization", equalized_image)

        # Apply and display histogram matching if reference image is available
        if reference_image is not None:
            matched_image = apply_histogram_matching(source_image, reference_image)
            display_image("Histogram Matching", matched_image)

        # Apply and display contrast stretching
        contrast_stretched_image = apply_contrast_stretching(source_image)
        display_image("Contrast Stretching", contrast_stretched_image)

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
