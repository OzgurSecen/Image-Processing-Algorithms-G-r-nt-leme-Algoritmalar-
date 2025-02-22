import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_lowpass_gaussian_filter(image, kernel_size):
    """Applies a Gaussian lowpass filter to the image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_median_filter(image, kernel_size):
    """Applies a median filter (non-linear) to the image."""
    return cv2.medianBlur(image, kernel_size)

def apply_weighted_smoothing_filter(image):
    """Applies a weighted smoothing filter to the image."""
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
    return cv2.filter2D(image, -1, kernel)

def apply_sharpening_filter(image):
    """Applies a basic sharpening filter to the image."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_laplacian(image):
    """Applies the Laplacian filter to the image."""
    return cv2.Laplacian(image, cv2.CV_64F)

def apply_first_derivative_filter(image):
    """Applies a first derivative filter (Sobel) to the image."""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(sobelx, sobely)

def apply_laplacian_enhancement(image):
    """Enhances the image using the Laplacian filter."""
    laplacian = apply_laplacian(image)
    # Convert the Laplacian result to uint8
    laplacian = cv2.convertScaleAbs(laplacian)
    enhanced_image = cv2.addWeighted(image, 1, laplacian, -1, 0)
    return enhanced_image


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

        # Apply and display Gaussian lowpass filter
        gaussian_image = apply_lowpass_gaussian_filter(image, kernel_size=5)
        display_image("Gaussian Lowpass Filter", gaussian_image)

        # Apply and display median filter
        median_image = apply_median_filter(image, kernel_size=5)
        display_image("Median Filter", median_image)

        # Apply and display weighted smoothing filter
        weighted_image = apply_weighted_smoothing_filter(image)
        display_image("Weighted Smoothing Filter", weighted_image)

        # Apply and display sharpening filter
        sharpened_image = apply_sharpening_filter(image)
        display_image("Sharpening Filter", sharpened_image)

        # Apply and display Laplacian filter
        laplacian_image = apply_laplacian(image)
        display_image("Laplacian Filter", laplacian_image)

        # Apply and display first derivative filter (Sobel)
        sobel_image = apply_first_derivative_filter(image)
        display_image("First Derivative Filter (Sobel)", sobel_image)

        # Apply and display Laplacian enhancement
        laplacian_enhanced_image = apply_laplacian_enhancement(image)
        display_image("Laplacian Image Enhancement", laplacian_enhanced_image)

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
