import os
import subprocess
import argparse
import cv2
import math

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
    
    if extract_frames_with_opencv(video_path, output_dir, fps):
        return

    # Fallback ffmpeg command to extract frames
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


def extract_frames_with_opencv(video_path, output_dir, fps=1):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("OpenCV could not open the video. Trying ffmpeg fallback.")
        return False

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not source_fps or math.isnan(source_fps) or source_fps <= 0:
        source_fps = 30

    frame_interval = max(1, int(round(source_fps / max(fps, 1))))
    saved_count = 0
    frame_index = 0

    print(f"Extracting frames from {video_path} at {fps} fps into {output_dir}")
    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % frame_interval == 0:
            saved_count += 1
            output_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(output_path, frame)

        frame_index += 1

    capture.release()
    print(f"Frame extraction completed. Saved {saved_count} frames.")
    return saved_count > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video using ffmpeg")
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save frames')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second')
    args = parser.parse_args()

    extract_frames(args.video_path, args.output_dir, args.fps)
