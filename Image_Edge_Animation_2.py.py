import cv2
import numpy as np
import os
import random
import imageio.v2 as iio


def create_strobe_edge_video():
    """
    Guides the user to create dynamic edge images, select video parameters (duration, FPS, BG color, output format, thickness),
    and then generates a video (MP4 or GIF) from the selected set of frames, using a dynamic, incremental naming scheme.
    """

    # --- 1. User Input and Configuration ---

    # Initialize variables to prevent NameError on early exit
    duration_s = 0.0
    fps = 0
    num_frames = 0
    bg_choice = ""
    output_format = ""
    thickness_mult = 1.0

    # Get image path
    input_image_path = input("Enter the path to your input image (e.g., 'photo.jpg'): ")
    if not os.path.exists(input_image_path):
        print(f"Error: Image not found at {input_image_path}")
        # Simple test image creation (optional, for demonstration if file is missing)
        if os.path.basename(input_image_path) == 'test_image.png':
            print("Creating a placeholder 'test_image.png'...")
            img = np.zeros((400, 400, 3), dtype=np.uint8)
            img[:] = (200, 200, 200)
            cv2.circle(img, (200, 200), 100, (0, 0, 255), -1)
            cv2.imwrite(input_image_path, img)
        else:
            return  # Exit if file not found and test image not created

    # Get video duration
    while True:
        try:
            duration_s = float(input("Enter desired video duration in seconds (e.g., 2.5): "))
            if duration_s <= 0:
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter a positive number for duration.")

    # Get FPS (must be an integer)
    while True:
        try:
            fps = int(input("Enter desired video FPS (e.g., 30 or 60): "))
            if fps <= 0:
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter a positive integer for FPS.")

    # CALCULATE num_frames immediately after getting duration and FPS
    num_frames = int(duration_s * fps)
    if num_frames == 0:
        print("Error: Calculated number of frames is zero. Check duration and FPS inputs.")
        return

    # Get background preference
    while True:
        bg_choice = input("Choose background color ('black' or 'white'): ").lower()
        if bg_choice in ['black', 'white']:
            break
        print("Invalid choice. Please enter 'black' or 'white'.")

    # Get output format preference
    while True:
        output_format = input("Choose output format ('mp4' or 'gif'): ").lower()
        if output_format in ['mp4', 'gif']:
            break
        print("Invalid choice. Please enter 'mp4' or 'gif'.")

    # Get line thickness multiplier
    while True:
        try:
            thickness_mult = float(input("Enter line thickness multiplier (1.0 for normal, 1.5 for 50% thicker): "))
            if thickness_mult < 1.0:
                print("Multiplier should be 1.0 or greater.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a number (e.g., 1.0, 1.5, 2.0).")

    # --- 2. Dynamic Naming Scheme ---

    # Extract base filename (without extension)
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]

    # Calculate base thickness for naming
    thickness_name = f'x{thickness_mult:.1f}'

    # Naming components
    animation_type = 'strobe'
    colour = bg_choice
    length_str = f'{duration_s:.1f}s'
    fps_str = f'{fps}fps'

    # Base name for folder/video
    base_name_full = f'{base_name}_{animation_type}_{colour}_bg_{length_str}_{fps_str}_{thickness_name}'

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

    # --- 3. Image Processing and Frame Generation ---

    print(f"\n--- Generating {num_frames} Dynamic Frames ---")

    # Calculate Dilation strength based on multiplier
    base_kernel_size = 3
    dilation_kernel_size = max(1, int(round(base_kernel_size * thickness_mult)))
    if dilation_kernel_size % 2 == 0:
        dilation_kernel_size += 1

    print(f"Using Dilation Kernel Size: {dilation_kernel_size}x{dilation_kernel_size}")

    # Define the kernel for Dilation
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)

    img = cv2.imread(input_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny parameters
    low_thresh_base = 80
    high_thresh_base = 180
    thresh_range = 40

    for i in range(num_frames):
        # Random Jitter
        noise = np.random.randint(-10, 10, gray.shape, dtype=np.int8)
        jittered_gray = cv2.add(gray, noise, dtype=cv2.CV_8U)

        # Canny Thresholds with variation (loops every 30 frames)
        cycle_i = i % 30
        offset = int(thresh_range * (cycle_i / 29) - thresh_range / 2)
        low_thresh = low_thresh_base + offset
        high_thresh = high_thresh_base + offset

        if low_thresh < 10: low_thresh = 10
        if high_thresh <= low_thresh: high_thresh = low_thresh + 10

        # Apply Canny
        edges = cv2.Canny(jittered_gray, low_thresh, high_thresh)

        # Apply Dilation to increase line thickness
        if thickness_mult > 1.0:
            edges = cv2.dilate(edges, kernel, iterations=1)

        # Select background
        if bg_choice == 'black':
            final_frame = edges  # White outlines on black
        else:
            final_frame = cv2.bitwise_not(edges)  # Black outlines on white

        # Save frame
        padding = len(str(num_frames - 1))
        filename = f'frame_{i:0{padding}d}_{bg_choice}_bg.png'
        cv2.imwrite(os.path.join(output_dir, filename), final_frame)

    print(f"Successfully created {num_frames} dynamic frames in '{output_dir}'.")

    # --- 4. Video/GIF Creation ---

    print(f"\n--- Creating {output_format.upper()} Output ---")

    # Get a list of frame paths
    frame_files = [os.path.join(output_dir, f'frame_{i:0{padding}d}_{bg_choice}_bg.png') for i in range(num_frames)]

    if output_format == 'mp4':
        # MP4 Creation (using OpenCV)
        first_frame = cv2.imread(frame_files[0])
        height, width, layers = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

        for frame_path in frame_files:
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
    print(f"Video duration: {duration_s} seconds at {fps} FPS.")


if __name__ == '__main__':
    # Dependencies: pip install opencv-python numpy imageio
    create_strobe_edge_video()