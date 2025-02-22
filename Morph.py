import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_morphological_operations(image, operation, kernel_size=5):
    """Applies morphological operations to the image."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if operation == "erosion":
        result = cv2.erode(image, kernel, iterations=1)
    elif operation == "dilation":
        result = cv2.dilate(image, kernel, iterations=1)
    elif operation == "opening":
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == "closing":
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError("Unsupported operation. Use 'erosion', 'dilation', 'opening', or 'closing'.")
    return result

def apply_hough_transform(image):
    """Applies Hough Transform to detect lines in the image."""
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return line_image

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

        # Threshold the image to create a binary image for morphological operations
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        display_image("Binary Image", binary_image)

        # Morphological Filtering (Erosion, Dilation, Opening, Closing)
        for operation in ["erosion", "dilation", "opening", "closing"]:
            result = apply_morphological_operations(binary_image, operation)
            display_image(f"Morphological Operation: {operation.capitalize()}", result)

        # Hough Transform
        hough_result = apply_hough_transform(image)
        display_image("Hough Transform (Line Detection)", hough_result)

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
