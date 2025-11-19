import cv2
import numpy as np
import os
import random
import imageio.v2 as iio


def create_feedback_loop_video():
    """
    Applies a pixelation and slight color-shift effect repeatedly to the previous frame's output,
    creating a continuous degradation video (feedback loop) until the image is unrecognizable.
    """

    # --- 1. User Input and Configuration ---

    # Initialize variables to prevent NameError on early exit
    duration_s = 0.0
    fps = 0
    num_frames = 0
    output_format = ""
    # We will repurpose thickness_mult for the decay factor
    decay_factor = 0.0

    # Get image path
    input_image_path = input("Enter the path to your input image (e.g., 'photo.jpg'): ")
    if not os.path.exists(input_image_path):
        print(f"Error: Image not found at {input_image_path}")
        # Simple test image creation (for demonstration if file is missing)
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

    # NEW: Get Decay Factor (how fast the image decays)
    while True:
        try:
            # Multiplier: 0.95 means image is 5% smaller/larger each cycle, 0.90 means 10%.
            decay_factor = float(input("Enter decay factor (0.95 for slow, 0.90 for fast decay/pixelation): "))
            if not (0.80 <= decay_factor < 1.0):
                print("Factor must be between 0.80 and 0.99 for a stable loop.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a number (e.g., 0.95).")

    # --- 2. Dynamic Naming Scheme ---

    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    animation_type = 'feedback'
    # Use the decay factor in the name
    decay_name = f'decay{decay_factor:.2f}'
    length_str = f'{duration_s:.1f}s'
    fps_str = f'{fps}fps'

    # Base name for folder/video
    base_name_full = f'{base_name}_{animation_type}_{decay_name}_{length_str}_{fps_str}'

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

    # --- 3. Image Processing and Frame Generation (The Feedback Loop) ---

    print(f"\n--- Generating {num_frames} Feedback Frames ---")

    # Load the original image and set up initial state
    current_img = cv2.imread(input_image_path)
    if current_img is None:
        print("Fatal Error: Could not load image for processing.")
        return

    original_height, original_width = current_img.shape[:2]

    frame_files = []
    padding = len(str(num_frames - 1))

    for i in range(num_frames):

        # 1. Pixelation/Degradation (Shrink and Re-enlarge)
        # This is the core feedback effect applied to the previous frame's output
        new_width = int(current_img.shape[1] * decay_factor)
        new_height = int(current_img.shape[0] * decay_factor)

        # Ensure size doesn't drop to zero or become too small for the effect
        if new_width < 10 or new_height < 10:
            print(f"\nImage completely decayed at frame {i}. Stopping loop.")
            num_frames = i
            break

        # Shrink using INTER_NEAREST for blocky pixelation
        shrunken_img = cv2.resize(current_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        # Re-enlarge back to the original size
        current_img = cv2.resize(shrunken_img, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # 2. Optional: Subtle Color Shift (introduces psychedelic drift)
        # We shift the hue channel slightly for visual interest
        hsv_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2HSV)
        hsv_img[:, :, 0] = (hsv_img[:, :, 0] + 1) % 180  # Shift Hue by 1 (HSV max is 180)
        current_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        # Save frame
        filename = f'frame_{i:0{padding}d}.png'
        frame_path = os.path.join(output_dir, filename)
        cv2.imwrite(frame_path, current_img)
        frame_files.append(frame_path)

    print(f"Successfully created {len(frame_files)} dynamic frames in '{output_dir}'.")

    # --- 4. Video/GIF Creation ---

    print(f"\n--- Creating {output_format.upper()} Output ---")

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

                # We pass the actual FPS here
        iio.mimsave(video_filename, gif_frames, fps=fps)

    print(f"File creation complete! 🎉")
    print(f"Output folder: {os.path.abspath(output_dir)}")
    print(f"Output file: {os.path.abspath(video_filename)}")
    actual_duration = len(frame_files) / fps
    print(f"Video duration: {actual_duration:.2f} seconds at {fps} FPS.")


if __name__ == '__main__':
    # Dependencies: pip install opencv-python numpy imageio
    create_feedback_loop_video()