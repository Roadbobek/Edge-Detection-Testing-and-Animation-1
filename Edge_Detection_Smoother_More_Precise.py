import cv2
import numpy as np
import os

def process_image_edges(input_path):
    """
    Takes an image, detects edges, and saves 20 output images in seperate directories (10 black on white, 10 white on black).

    Args:
        input_path (str): The path to the input image.
    """
    # Create the Black BG and White BG output directory if it doesn't exist
    if not os.path.exists('Edge_Detection_Smoother_More_Precise_Output_Black_BG'):
        os.makedirs('Edge_Detection_Smoother_More_Precise_Output_Black_BG')
    if not os.path.exists('Edge_Detection_Smoother_More_Precise_Output_White_BG'):
        os.makedirs('Edge_Detection_Smoother_More_Precise_Output_White_BG')

    # Load the image
    img = cv2.imread(input_path)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not load image from {input_path}")
        return

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and help with edge detection
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred_img, 100, 200)

    # Create two types of output images
    for i in range(1, 11):
        # 1. White outlines on a black background
        black_bg = np.zeros_like(edges)
        white_on_black = cv2.bitwise_or(black_bg, edges)
        cv2.imwrite(f'Edge_Detection_Smoother_More_Precise_Output_Black_BG/white_on_black_edge_{i}.png', white_on_black)

        # 2. Black outlines on a white background
        white_bg = np.ones_like(edges) * 255
        inverted_edges = cv2.bitwise_not(edges)
        black_on_white = cv2.bitwise_and(white_bg, inverted_edges)
        cv2.imwrite(f'Edge_Detection_Smoother_More_Precise_Output_White_BG/black_on_white_edge_{i}.png', black_on_white)

    print("Image processing complete! Check the 'Edge_Detection_Smoother_More_Precise_Output_Black_BG' and 'Edge_Detection_Smoother_More_Precise_Output_White_BG' folders for the results. ✅")

if __name__ == "__main__":
    # Replace 'input_image.png' with the path to your image file
    # Ensure the image is in the same directory as the script or provide the full path
    process_image_edges('input_image.png')