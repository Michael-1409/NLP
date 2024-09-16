import cv2
import numpy as np

# 1. Load an image
def load_image(image_path):
    image = cv2.imread(image_path)  # Load image in color (default)
    if image is None:
        print("Error: Unable to load image.")
        return None
    return image

# 2. Convert image to grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Detect edges using Canny edge detection
def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

# 4. Display the image in a window
def display_image(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()  # Close the window after key press

# 5. Save the processed image to disk
def save_image(output_path, image):
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

# Example usage
if __name__ == "__main__":
    image_path = 'input_image.jpg'  # Replace with your image path
    output_path = 'output_image.jpg'
    
    # Load and display the original image
    image = load_image(image_path)
    if image is not None:
        display_image("Original Image", image)
        
        # Convert to grayscale
        gray_image = convert_to_grayscale(image)
        display_image("Grayscale Image", gray_image)
        
        # Detect edges
        edges_image = detect_edges(gray_image)
        display_image("Edges Detected", edges_image)
        
        # Save the edges image
        save_image(output_path, edges_image)
