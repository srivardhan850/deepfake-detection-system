# Preprocessing Scripts for Deepfake Detection Project

This directory contains preprocessing scripts to prepare datasets for training deepfake detection models.

## Scripts

### 1. Frame extraction

`extract_frames.py`  
Extracts frames from input videos at a configurable frames-per-second (FPS) rate using ffmpeg.

Usage example:
```
python extract_frames.py --video_path /path/to/video.mp4 --output_dir /path/to/frames --fps 2
```

### 2. Face detection and alignment

`face_align_mtcnn.py`  
Detects and aligns faces in the extracted frames using MTCNN (from the facenet-pytorch library). Saves aligned face crops as 224x224 images.

Usage example:
```
python face_align_mtcnn.py --frame_dir /path/to/frames --output_dir /path/to/aligned_faces
```

## Dataset Folder Structure

Recommended folder structure for training data:

```
dataset/
  real/
    video1/
      aligned_faces/
        frame_000001_face1.jpg
        frame_000002_face1.jpg
        ...
    video2/
      aligned_faces/
        ...
  fake/
    videoX/
      aligned_faces/
        ...
    videoY/
      aligned_faces/
        ...
```

Each video directory contains aligned face crops extracted from its frames to facilitate training frame-level classifiers.

You can adapt the scripts or folder layout as needed.

## Requirements

- Python 3.7+
- ffmpeg installed and in PATH
- PyTorch and facenet-pytorch library (`pip install torch facenet-pytorch`)

## Next steps

After preprocessing, use the aligned faces dataset for training with the provided training notebook.
