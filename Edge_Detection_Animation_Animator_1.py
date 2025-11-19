import cv2
import os


def create_video_from_frames(input_dir, output_video_name, fps=60):
    """
    Creates a video file from sequenced image frames.

    Args:
        input_dir (str): Directory containing the sequential image frames.
        output_video_name (str): The name of the resulting video file (e.g., '1_second_edge_loop.mp4').
        fps (int): Frames per second for the output video.
    """
    # 1. Define the input frames sequence
    # We need to list the files in the correct sequential order (00, 01, 02, etc.)
    # The glob module is helpful for this, but simple list comprehension and sorting is also effective.

    frame_files = []
    # Loop through the 30 unique frames (0 to 29)
    for i in range(30):
        # Format the frame index with leading zeros (e.g., '00', '01')
        frame_id = f'{i:02d}'

        # Add the black background version
        frame_files.append(os.path.join(input_dir, f'frame_{frame_id}_black_bg.png'))

        # Add the white background version
        frame_files.append(os.path.join(input_dir, f'frame_{frame_id}_white_bg.png'))

    # Check if we found 60 frames
    if len(frame_files) != 60:
        print(f"Error: Found {len(frame_files)} frames. Expected 60.")
        print("Please check the input directory name and image file names.")
        return

    # 2. Read the first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"Error: Could not read the first frame at {frame_files[0]}.")
        return

    height, width, layers = first_frame.shape

    # 3. Define the video writer
    # The 'mp4v' codec is commonly supported and suitable for MP4 container.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    print(f"Creating video '{output_video_name}' at {fps} FPS...")

    # 4. Write all frames to the video file
    for filename in frame_files:
        img = cv2.imread(filename)
        # Ensure the image is successfully loaded before writing
        if img is not None:
            video.write(img)

    # 5. Release the video writer object
    video.release()

    print("Video creation complete!")
    print(f"Saved video to: {os.path.abspath(output_video_name)}")


if __name__ == '__main__':
    # --- Configuration ---
    input_directory = 'dynamic_edges_output'  # Must match the output folder from the previous script
    output_video_file = '1_second_60fps_edge_video.mp4'
    target_fps = 60  # 60 frames (images) / 60 FPS = 1 second video

    # Run the function
    create_video_from_frames(input_directory, output_video_file, target_fps)