import cv2
import numpy as np
import os


def create_dynamic_edges(input_image_path, output_dir, num_frames=30):
    """
    Generates a set of images with varying edge details (30 white-bg, 30 black-bg)
    by systematically adjusting the Canny edge detection thresholds.

    Args:
        input_image_path (str): The file path of the input image.
        output_dir (str): The directory to save all 60 images.
        num_frames (int): The number of unique frames (30 pairs = 60 images total).
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read the image
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Could not read image at {input_image_path}")
        return

    # Convert to grayscale and apply Gaussian Blur for cleaner results
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray, (5, 5), 0)

    # Define a range for the Canny thresholds to create variation
    # Canny thresholds: (low_threshold, high_threshold)

    # We'll vary the lower threshold from 50 up to 150
    # And the higher threshold from 150 up to 250
    low_start = 50
    low_end = 150
    high_start = 150
    high_end = 250

    # Calculate the step for each threshold over the number of frames
    low_step = (low_end - low_start) / (num_frames - 1)
    high_step = (high_end - high_start) / (num_frames - 1)

    print(f"Generating {num_frames} unique edge variations...")

    for i in range(num_frames):
        # Calculate the current thresholds
        low_thresh = int(low_start + i * low_step)
        high_thresh = int(high_start + i * high_step)

        # Ensure high_thresh is always greater than low_thresh
        if low_thresh >= high_thresh:
            high_thresh = low_thresh + 10  # Maintain a minimum difference

        # Apply Canny edge detection with the varying thresholds
        edges = cv2.Canny(blurred_img, low_thresh, high_thresh)

        # 1. White outlines on a black background (edges)
        cv2.imwrite(os.path.join(output_dir, f'frame_{i:02d}_black_bg.png'), edges)

        # 2. Black outlines on a white background (inverted edges)
        edges_inverted = cv2.bitwise_not(edges)
        cv2.imwrite(os.path.join(output_dir, f'frame_{i:02d}_white_bg.png'), edges_inverted)

    print(f"Successfully created {num_frames * 2} images in the '{output_dir}' directory.")
    print("These 60 images are sequenced to create a dynamic 1-second animation (60 FPS).")


# --- Example Usage ---
if __name__ == '__main__':
    # You must replace this with the path to your actual image file (e.g., 'photo.jpg')
    # For demonstration, we'll create a placeholder image path.
    input_file_path = 'input_image_2.png'
    output_directory = 'dynamic_edges_output'

    # --- DEMO: Create a test image if it doesn't exist ---
    if not os.path.exists(input_file_path):
        # Simple placeholder image (rectangle on a plain background)
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img[:] = (200, 200, 200)  # Light gray background
        cv2.rectangle(img, (50, 50), (350, 350), (50, 50, 50), 20)  # Darker gray border
        cv2.circle(img, (200, 200), 100, (0, 0, 255), -1)  # Red circle
        cv2.imwrite(input_file_path, img)
        print(f"Created a test image: {input_file_path}")
    # ----------------------------------------------------

    # Run the function to generate the frames
    create_dynamic_edges(input_file_path, output_directory, num_frames=30)