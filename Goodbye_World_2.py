import cv2
import numpy as np
import os
import random
import imageio.v2 as iio


def create_feedback_loop_video():
    """
    Applies a Gaussian Blur and slight color-shift effect repeatedly to the previous frame's output,
    creating a continuous, abstract degradation video (feedback loop).
    """

    # --- 1. User Input and Configuration ---

    # Initialize variables to prevent NameError on early exit
    duration_s = 0.0
    fps = 0
    num_frames = 0
    output_format = ""
    # We will repurpose the factor for blur strength
    blur_strength = 0.0

    # Get image path
    input_image_path = input("Enter the path to your input image (e.g., 'photo.jpg'): ")
    if not os.path.exists(input_image_path):
        print(f"Error: Image not found at {input_image_path}")
        if os.path.basename(input_image_path) == 'test_image.png':
            print("Creating a placeholder 'test_image.png'...")
            img = np.zeros((400, 400, 3), dtype=np.uint8)
            img[:] = (255, 100, 0)  # Orange background
            cv2.circle(img, (200, 200), 100, (255, 255, 255), -1)
            cv2.imwrite(input_image_path, img)
        else:
            return

    # Get video duration
    while True:
        try:
            duration_s = float(input("Enter desired video duration in seconds (e.g., 3.0): "))
            if duration_s <= 0:
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter a positive number for duration.")

    # Get FPS
    while True:
        try:
            fps = int(input("Enter desired video FPS (e.g., 20 or 30): "))
            if fps <= 0:
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter a positive integer for FPS.")

    # CALCULATE num_frames
    num_frames = int(duration_s * fps)
    if num_frames == 0:
        print("Error: Calculated number of frames is zero. Check duration and FPS inputs.")
        return

    # Get output format preference
    while True:
        output_format = input("Choose output format ('mp4' or 'gif'): ").lower()
        if output_format in ['mp4', 'gif']:
            break
        print("Invalid choice. Please enter 'mp4' or 'gif'.")

    # NEW: Get Blur Factor (Gaussian Kernel Size)
    while True:
        try:
            # Factor: Use small odd numbers for slow decay (3, 5, 7)
            blur_strength = int(
                input("Enter blur strength (Small odd number, e.g., 3 for slow, 7 for fast degradation): "))
            if blur_strength < 3 or blur_strength % 2 == 0:
                print("Strength must be an odd integer 3 or greater.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # --- 2. Dynamic Naming Scheme ---

    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    animation_type = 'blur_feedback'
    # Use the blur strength in the name
    blur_name = f'blur{blur_strength}'
    length_str = f'{duration_s:.1f}s'
    fps_str = f'{fps}fps'

    # Base name for folder/video
    base_name_full = f'{base_name}_{animation_type}_{blur_name}_{length_str}_{fps_str}'

    # Incrementing Folder Name
    increment = 1
    output_dir = f'{base_name_full}_i{increment}'
    while os.path.exists(output_dir):
        increment += 1
        output_dir = f'{base_name_full}_i{increment}'

    # Final video filename
    video_filename = f'{base_name_full}_i{increment}.{output_format}'

    # Create the final directory
    os.makedirs(output_dir, exist_ok=False)

    # --- 3. Image Processing and Frame Generation (The Blur Feedback Loop) ---

    print(f"\n--- Generating {num_frames} Abstract Frames ---")

    # Load the original image and set up initial state
    current_img = cv2.imread(input_image_path)
    if current_img is None:
        print("Fatal Error: Could not load image for processing.")
        return

    original_height, original_width = current_img.shape[:2]

    frame_files = []
    padding = len(str(num_frames - 1))

    # Define the Gaussian Kernel size (K) based on user input
    kernel_size = (blur_strength, blur_strength)

    print(f"Using Gaussian Kernel Size: {blur_strength}x{blur_strength}")

    for i in range(num_frames):
        # 1. Apply Gaussian Blur
        # The result (current_img) immediately becomes the input for the next frame.
        current_img = cv2.GaussianBlur(current_img, kernel_size, 0)

        # 2. Optional: Subtle Color Shift (introduces psychedelic drift)
        # We shift the hue channel slightly for visual interest
        hsv_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2HSV)
        # Shift Hue by 1. The cumulative effect over many frames makes colors drift.
        hsv_img[:, :, 0] = (hsv_img[:, :, 0] + 1) % 180
        current_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        # 3. Save frame
        filename = f'frame_{i:0{padding}d}.png'
        frame_path = os.path.join(output_dir, filename)
        cv2.imwrite(frame_path, current_img)
        frame_files.append(frame_path)

    print(f"Successfully created {num_frames} dynamic frames in '{output_dir}'.")

    # --- 4. Video/GIF Creation ---

    print(f"\n--- Creating {output_format.upper()} Output ---")

    if output_format == 'mp4':
        # MP4 Creation (using OpenCV)
        first_frame = cv2.imread(frame_files[0])
        height, width, layers = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

        for frame_path in fr/ame_files:
            img = cv2.imread(frame_path)
            if img is not None:
                video.write(img)
        video.release()

    elif output_format == 'gif':
        # GIF Creation (using imageio)

        gif_frames = []
        for frame_path in frame_files:
            img = cv2.imread(frame_path)
            if img is not None:
                gif_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        iio.mimsave(video_filename, gif_frames, fps=fps)

    print(f"File creation complete! 🎉")
    print(f"Output folder: {os.path.abspath(output_dir)}")
    print(f"Output file: {os.path.abspath(video_filename)}")
    actual_duration = num_frames / fps
    print(f"Video duration: {actual_duration:.2f} seconds at {fps} FPS.")


if __name__ == '__main__':
    # Dependencies: pip install opencv-python numpy imageio
    create_feedback_loop_video()