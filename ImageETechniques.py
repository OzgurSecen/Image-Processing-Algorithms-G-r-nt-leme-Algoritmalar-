import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_negative(image):
    """Applies negative transformation to the image."""
    return cv2.bitwise_not(image)

def apply_thresholding(image, threshold=128):
    """Applies thresholding to the image."""
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def apply_logarithmic_transformation(image):
    """Applies logarithmic transformation to the image."""
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(1 + image))
    return np.uint8(log_image)

def apply_power_law_transformation(image, gamma):
    """Applies power law (gamma) transformation to the image."""
    normalized_image = image / 255.0
    power_image = np.power(normalized_image, gamma) * 255
    return np.uint8(power_image)

def apply_bit_plane_slicing(image, plane):
    """Extracts a specific bit plane from the image."""
    return cv2.bitwise_and(image, (1 << plane)) * 255

def apply_linear_filter(image, kernel):
    """Applies a linear spatial filter to the image."""
    return cv2.filter2D(image, -1, kernel)

def apply_median_filter(image, kernel_size):
    """Applies a nonlinear median filter to the image."""
    return cv2.medianBlur(image, kernel_size)

def display_image(title, image):
    """Displays an image using Matplotlib."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()

def main():
    # Input Image Path
    image_path = "resim.jpg"  # Replace with your image path

    try:
        # Load the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image not found. Please check the path.")

        # Apply and display negative image
        negative_image = apply_negative(image)
        display_image("Negative Image", negative_image)

        # Apply and display thresholding
        thresholded_image = apply_thresholding(image)
        display_image("Thresholded Image", thresholded_image)

        # Apply and display logarithmic transformation
        log_image = apply_logarithmic_transformation(image)
        display_image("Logarithmic Transformation", log_image)

        # Apply and display power law transformation
        power_image = apply_power_law_transformation(image, gamma=2.0)
        display_image("Power Law Transformation (Gamma=2.0)", power_image)

        # Apply and display bit plane slicing
        bit_plane_image = apply_bit_plane_slicing(image, plane=3)
        display_image("Bit Plane Slicing (Plane=3)", bit_plane_image)

        # Apply and display linear filter
        kernel = np.ones((3, 3), np.float32) / 9  # Example kernel
        linear_filtered_image = apply_linear_filter(image, kernel)
        display_image("Linear Filtered Image", linear_filtered_image)

        # Apply and display median filter
        median_filtered_image = apply_median_filter(image, kernel_size=3)
        display_image("Median Filtered Image", median_filtered_image)

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
