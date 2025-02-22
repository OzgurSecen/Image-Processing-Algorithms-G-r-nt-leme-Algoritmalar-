import cv2
import numpy as np
import matplotlib.pyplot as plt

def local_edge_linking(image, threshold1, threshold2):
    """Performs edge linking using local processing (Canny edge detection)."""
    edges = cv2.Canny(image, threshold1, threshold2)
    return edges

def global_edge_linking_region_based(image):
    """Performs edge linking using region-based processing."""
    # Apply Canny edge detection
    edges = cv2.Canny(image, 50, 150)

    # Find contours in the edge-detected image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to visualize region-based linking
    linked_image = np.zeros_like(image)

    # Fill the contours to simulate region-based linking
    for contour in contours:
        cv2.drawContours(linked_image, [contour], -1, 255, thickness=cv2.FILLED)

    return linked_image

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

        # Local Edge Linking using Canny
        local_edges = local_edge_linking(image, threshold1=50, threshold2=150)
        display_image("Local Edge Linking (Canny)", local_edges)

        # Global Edge Linking using Region-Based Contour Filling
        region_based_edges = global_edge_linking_region_based(image)
        display_image("Global Edge Linking (Region-Based Contours)", region_based_edges)

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
