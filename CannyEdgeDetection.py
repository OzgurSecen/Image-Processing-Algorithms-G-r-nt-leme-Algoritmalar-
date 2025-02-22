import cv2
import matplotlib.pyplot as plt

def apply_canny_edge_detection(image, threshold1, threshold2):
    """Applies Canny Edge Detection to the input image."""
    edges = cv2.Canny(image, threshold1, threshold2)
    return edges

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

        # Apply Canny Edge Detection
        threshold1 = 50  # Lower threshold
        threshold2 = 150  # Upper threshold
        edges = apply_canny_edge_detection(image, threshold1, threshold2)

        # Display the original and edge-detected images
        display_image("Original Image", image)
        display_image("Canny Edge Detection", edges)

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
