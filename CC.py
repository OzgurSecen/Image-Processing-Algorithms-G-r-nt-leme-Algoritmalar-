import cv2
import numpy as np
import matplotlib.pyplot as plt

def connected_components(image):
    """Finds connected components in a binary image."""
    num_labels, labels = cv2.connectedComponents(image)
    return num_labels, labels

def connected_components_with_stats(image):
    """Finds connected components with stats in a binary image."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    return num_labels, labels, stats, centroids

def contour_detection(image):
    """Finds contours in a binary image."""
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

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

        # Threshold the image to create a binary image
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        display_image("Binary Image", binary_image)

        # Connected Components
        num_labels, labels = connected_components(binary_image)
        labeled_image = (labels * 255 / np.max(labels)).astype(np.uint8)
        display_image("Connected Components", labeled_image)

        # Connected Components with Stats
        num_labels, labels, stats, centroids = connected_components_with_stats(binary_image)
        print(f"Number of Labels: {num_labels}")
        print(f"Stats: {stats}")
        print(f"Centroids: {centroids}")

        # Contour Detection
        contours, hierarchy = contour_detection(binary_image)
        contour_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        display_image("Contour Detection", contour_image)

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
