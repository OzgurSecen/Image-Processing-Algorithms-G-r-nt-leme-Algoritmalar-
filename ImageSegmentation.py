import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_edge_detection(image, method="sobel"):
    """Applies edge detection using Roberts, Prewitt, or Sobel methods."""
    if method == "roberts":
        kernel_x = np.array([[1, 0], [0, -1]])
        kernel_y = np.array([[0, 1], [-1, 0]])
        edge_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
        edge_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    elif method == "prewitt":
        kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        edge_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
        edge_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    elif method == "sobel":
        edge_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        edge_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    else:
        raise ValueError("Unsupported method. Use 'roberts', 'prewitt', or 'sobel'.")

    magnitude = cv2.magnitude(edge_x, edge_y)
    return np.uint8(magnitude)


def apply_thresholding(image, method="global", block_size=11, c=2):
    """Applies global, adaptive, or other thresholding techniques."""
    if method == "global":
        _, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    elif method == "adaptive_mean":
        thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, block_size, c)
    elif method == "adaptive_gaussian":
        thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, block_size, c)
    else:
        raise ValueError("Unsupported method. Use 'global', 'adaptive_mean', or 'adaptive_gaussian'.")

    return thresholded

def apply_marr_hildreth(image):
    """Applies the Marr-Hildreth edge detection method."""
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    return np.uint8(np.absolute(laplacian))

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

        # Edge Detection (Roberts, Prewitt, Sobel)
        for method in ["roberts", "prewitt", "sobel"]:
            edges = apply_edge_detection(image, method=method)
            display_image(f"Edge Detection ({method.capitalize()})", edges)

        # Thresholding (Global, Adaptive Mean, Adaptive Gaussian)
        for method in ["global", "adaptive_mean", "adaptive_gaussian"]:
            thresholded = apply_thresholding(image, method=method)
            display_image(f"Thresholding ({method.replace('_', ' ').capitalize()})", thresholded)

        # Marr-Hildreth Edge Detection
        marr_hildreth_edges = apply_marr_hildreth(image)
        display_image("Marr-Hildreth Edge Detection", marr_hildreth_edges)

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
