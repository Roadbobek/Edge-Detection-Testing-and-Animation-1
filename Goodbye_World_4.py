import cv2
import numpy as np
import os
import random
import imageio.v2 as iio


def create_feedback_loop_video():
    """
    Applies a stable, aggressive block-pixelation effect repeatedly until the image is unrecognizable.
    """

    # --- 1. User Input and Configuration ---

    # Initialize variables to prevent NameError on early exit
    duration_s = 0.0
    fps = 0
    num_frames = 0
    output_format = ""
    # This will now be the base speed of block size increase
    pixel_increase_rate = 0.0

    # Get image path
    input_image_path = input("Enter the path to your input image (e.g., 'photo.jpg'): ")
    if not os.path.exists(input_image_path):
        print(f"Error: Image not found at {input_image_path}")
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

    # NEW: Get Pixelation Rate (how fast the block size increases)
    while True:
        try:
            # Factor: The number of pixels to add to the block size every second.
            pixel_increase_rate = float(
                input("Enter pixel block size increase rate (e.g., 20.0 for fast, 5.0 for slow): "))
            if pixel_increase_rate <= 0:
                print("Rate must be a positive number.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a number (e.g., 10.0).")

    # --- 2. Dynamic Naming Scheme ---

    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    animation_type = 'block_collapse'
    rate_name = f'rate{pixel_increase_rate:.1f}'
    length_str = f'{duration_s:.1f}s'
    fps_str = f'{fps}fps'

    # Base name for folder/video
    base_name_full = f'{base_name}_{animation_type}_{rate_name}_{length_str}_{fps_str}'

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

    # --- 3. Image Processing and Frame Generation (The Fixed Pixelation Loop) ---

    print(f"\n--- Generating {num_frames} Block Collapse Frames ---")

    # Load the original image once
    original_img = cv2.imread(input_image_path)
    if original_img is None:
        print("Fatal Error: Could not load image for processing.")
        return

    original_height, original_width = original_img.shape[:2]

    frame_files = []
    padding = len(str(num_frames - 1))

    # Calculate the total pixel increase over the whole video duration
    total_pixel_increase = pixel_increase_rate * duration_s

    for i in range(num_frames):

        # Calculate the current pixel block size
        # Start at 1 (no pixelation) and increase linearly over the video duration.
        current_increase = total_pixel_increase * (i / num_frames)
        # Block size must be an integer > 0
        pixel_block_size = max(1, int(1 + current_increase))

        # --- The Fixed Block-Pixelation Effect ---

        # 1. Shrink the original image down to the size of the pixel grid
        # The new dimensions are the original dimensions divided by the block size.
        shrink_w = int(original_width / pixel_block_size)
        shrink_h = int(original_height / pixel_block_size)

        # Ensure image is not too small (though the effect should work down to 1x1)
        if shrink_w < 1 or shrink_h < 1:
            print(f"\nImage has collapsed to a single pixel at frame {i}. Stopping loop.")
            num_frames = i
            break

        # Use INTER_NEAREST for the shrinkage to average blocks for the pixel color
        shrunken_img = cv2.resize(original_img, (shrink_w, shrink_h), interpolation=cv2.INTER_NEAREST)

        # 2. Re-enlarge back to the ORIGINAL dimensions
        # This makes the small grid visible as large, clean pixel blocks.
        final_frame = cv2.resize(shrunken_img, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # 3. Apply the Color Shift to the final frame (creates the drift)
        hsv_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        # Shift Hue based on the frame number for subtle color cycling
        # We'll use a slow shift to avoid a strobe effect
        color_shift = int(i * 0.5) % 180
        hsv_frame[:, :, 0] = (hsv_frame[:, :, 0] + color_shift) % 180
        current_img = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)

        # 4. Save frame
        filename = f'frame_{i:0{padding}d}.png'
        frame_path = os.path.join(output_dir, filename)
        cv2.imwrite(frame_path, current_img)
        frame_files.append(frame_path)

    print(f"Successfully created {len(frame_files)} dynamic frames in '{output_dir}'.")

    # --- 4. Video/GIF Creation ---

    print(f"\n--- Creating {output_format.upper()} Output ---")

    if output_format == 'mp4':
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
        gif_frames = []
        for frame_path in frame_files:
            img = cv2.imread(frame_path)
            if img is not None:
                gif_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        iio.mimsave(video_filename, gif_frames, fps=fps)

    print(f"File creation complete! 🎉")
    print(f"Output folder: {os.path.abspath(output_dir)}")
    print(f"Output file: {os.path.abspath(video_filename)}")
    actual_duration = len(frame_files) / fps
    print(f"Video duration: {actual_duration:.2f} seconds at {fps} FPS.")


if __name__ == '__main__':
    # Dependencies: pip install opencv-python numpy imageio
    create_feedback_loop_video()