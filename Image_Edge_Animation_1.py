import cv2
import numpy as np
import os
import random
import imageio.v2 as iio  # Import imageio for GIF creation


def create_strobe_edge_video():
    """
    Guides the user to create dynamic edge images, select video parameters (duration, FPS, BG color, output format),
    and then generates a video (MP4 or GIF) from the selected set of frames, using a dynamic, incremental naming scheme.
    """

    # --- 1. User Input and Configuration ---

    # Get image path
    input_image_path = input("Enter the path to your input image (e.g., 'photo.jpg'): ")
    if not os.path.exists(input_image_path):
        print(f"Error: Image not found at {input_image_path}")
        # Simple test image creation (optional, for demonstration if file is missing)
        if input_image_path == 'test_image.png':
            print("Creating a placeholder 'test_image.png'...")
            img = np.zeros((400, 400, 3), dtype=np.uint8)
            img[:] = (200, 200, 200)
            cv2.circle(img, (200, 200), 100, (0, 0, 255), -1)
            cv2.imwrite(input_image_path, img)
        else:
            return

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

    # Determine the total number of frames needed
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

    # --- 2. Dynamic Naming Scheme ---

    # Extract base filename (without extension)
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]

    # Naming components
    animation_type = 'strobe'
    colour = bg_choice
    length_str = f'{duration_s:.1f}s'
    fps_str = f'{fps}fps'

    # Base name for folder/video
    base_name_full = f'{base_name}_{animation_type}_{colour}_bg_{length_str}_{fps_str}'

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

    # --- 3. Image Processing and Frame Generation (Unchanged Logic) ---

    print(f"\n--- Generating {num_frames} Dynamic Frames ---")
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

        edges = cv2.Canny(jittered_gray, low_thresh, high_thresh)

        # Select background
        final_frame = edges if bg_choice == 'black' else cv2.bitwise_not(edges)

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

        # Read all images into a list
        gif_frames = []
        for frame_path in frame_files:
            img = cv2.imread(frame_path)
            # imageio needs BGR to be converted to RGB
            if img is not None:
                gif_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                # Save GIF. The duration parameter is 1/FPS (time per frame in seconds)
        iio.mimsave(video_filename, gif_frames, fps=fps)

    print(f"File creation complete! 🎉")
    print(f"Output folder: {os.path.abspath(output_dir)}")
    print(f"Output file: {os.path.abspath(video_filename)}")
    print(f"Video duration: {duration_s} seconds at {fps} FPS.")


if __name__ == '__main__':
    # You must install imageio for GIF support: pip install imageio
    create_strobe_edge_video()