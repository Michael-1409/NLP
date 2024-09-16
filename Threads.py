import cv2
import numpy as np
import threading

# 1. Load an image
def load_image(image_path):
    image = cv2.imread(image_path)  # Load image in color (default)
    if image is None:
        print("Error: Unable to load image.")
        return None
    return image

# 2. Convert image to grayscale
def convert_to_grayscale(image, result_dict):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result_dict['gray_image'] = gray_image  # Store result in a shared dictionary
    print("Grayscale conversion done.")

# 3. Detect edges using Canny edge detection
def detect_edges(image, result_dict, low_threshold=50, high_threshold=150):
    edges_image = cv2.Canny(image, low_threshold, high_threshold)
    result_dict['edges_image'] = edges_image  # Store result in a shared dictionary
    print("Edge detection done.")

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

        # Shared dictionary to store results from threads
        result_dict = {}

        # Create threads for parallel processing
        grayscale_thread = threading.Thread(target=convert_to_grayscale, args=(image, result_dict))
        edges_thread = threading.Thread(target=detect_edges, args=(image, result_dict))

        # Start the threads
        grayscale_thread.start()
        edges_thread.start()

        # Wait for both threads to complete
        grayscale_thread.join()
        edges_thread.join()

        # Access the results from the shared dictionary
        if 'gray_image' in result_dict:
            display_image("Grayscale Image", result_dict['gray_image'])
        if 'edges_image' in result_dict:
            display_image("Edges Detected", result_dict['edges_image'])
            save_image(output_path, result_dict['edges_image'])
