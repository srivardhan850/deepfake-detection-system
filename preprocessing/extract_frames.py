import os
import subprocess
import argparse

def extract_frames(video_path, output_dir, fps=1):
    """
    Extract frames from video at given fps using ffmpeg.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save extracted frames.
        fps (int): Frames per second to extract.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ffmpeg command to extract frames
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={fps}',
        os.path.join(output_dir, 'frame_%06d.jpg')
    ]
    
    print(f"Extracting frames from {video_path} at {fps} fps into {output_dir}")
    try:
        subprocess.run(command, check=True)
        print("Frame extraction completed.")
    except FileNotFoundError as e:
        print("Error: ffmpeg executable not found. Please make sure ffmpeg is installed and added to your system PATH.")
        raise e
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg command failed with error: {e}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video using ffmpeg")
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save frames')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second')
    args = parser.parse_args()

    extract_frames(args.video_path, args.output_dir, args.fps)
