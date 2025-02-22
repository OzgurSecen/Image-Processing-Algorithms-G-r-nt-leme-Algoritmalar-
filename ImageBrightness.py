import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_brightness(image_path, alpha):
    """
    Adjusts the brightness of an image.

    Parameters:
        image_path (str): Path to the input image.
        alpha (float): Brightness factor. (>1 increases brightness, <1 decreases brightness)

    Returns:
        bright_image (numpy.ndarray): Brightness adjusted image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found. Please check the path.")

    bright_image = cv2.convertScaleAbs(image, alpha=alpha)
    return bright_image

def display_image(title, image):
    """Displays an image using Matplotlib."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

def main():
    # Input Image Path
    image_path = "resim.jpg"  # Updated image path

    try:
        # Load and adjust brightness
        original_image = cv2.imread(image_path)

        # Display original image
        display_image("Original Image", original_image)

        # Increase brightness
        bright_image = adjust_brightness(image_path, alpha=1.5)
        display_image("Increased Brightness", bright_image)

        # Decrease brightness
        dark_image = adjust_brightness(image_path, alpha=0.5)
        display_image("Decreased Brightness", dark_image)

        # Image Sampling and Quantization Example
        small_image = cv2.resize(original_image, (original_image.shape[1] // 4, original_image.shape[0] // 4))
        display_image("Downsampled Image (Sampling)", small_image)

        # Interpolation example
        interpolated_image = cv2.resize(small_image, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        display_image("Interpolated Image", interpolated_image)

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
