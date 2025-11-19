import cv2
import numpy as np


def separate_edges(input_image_path, output_dir_black_bg, output_dir_white_bg, num_images=10):
    """
    Takes an input image, detects edges, and saves multiple copies
    of the edge images with both black and white backgrounds.

    Args:
        input_image_path (str): The file path of the input image.
        output_dir_black_bg (str): The directory to save images with white edges on a black background.
        output_dir_white_bg (str): The directory to save images with black edges on a white background.
        num_images (int): The number of copies to save for each type of image.
    """
    # Create output directories if they don't exist
    import os
    os.makedirs(output_dir_black_bg, exist_ok=True)
    os.makedirs(output_dir_white_bg, exist_ok=True)

    # Read the image in color
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Could not read image at {input_image_path}")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    # The two threshold values are crucial. A smaller value detects more edges,
    # and a larger value detects fewer, stronger edges.
    edges = cv2.Canny(gray, 100, 200)

    # Invert the edges to get black edges on a white background
    # The edges variable is a binary image (0s and 255s), so we can
    # invert it easily.
    edges_inverted = cv2.bitwise_not(edges)

    # Save the images
    for i in range(num_images):
        # Save white outlines on a black background
        cv2.imwrite(os.path.join(output_dir_black_bg, f'edges_black_bg_{i + 1}.png'), edges)

        # Save black outlines on a white background
        cv2.imwrite(os.path.join(output_dir_white_bg, f'edges_white_bg_{i + 1}.png'), edges_inverted)

    print(f"Successfully saved {num_images} images with a black background and "
          f"{num_images} images with a white background.")
    print(f"Images with black backgrounds saved in: {output_dir_black_bg}")
    print(f"Images with white backgrounds saved in: {output_dir_white_bg}")


# Example usage:
# Create a test image to use if you don't have one
# You'd typically replace this with the path to your own image
# Here we just create a simple red square on a blue background
def create_test_image(path='test_image.png'):
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)  # Blue background (BGR format)
    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 255), -1)  # Red square
    cv2.imwrite(path, img)
    return path


if __name__ == '__main__':
    # # Create a test image for demonstration
    # input_image_path = create_test_image()

    # Define image
    input_image_path = 'input_image.png'

    # Define output directories
    output_dir_black_bg = 'Edge_Detection_Aggressive_Output_Black_GB'
    output_dir_white_bg = 'Edge_Detection_Aggressive_Output_White_BG'

    # Run the function
    separate_edges(input_image_path, output_dir_black_bg, output_dir_white_bg)