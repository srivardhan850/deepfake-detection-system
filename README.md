# Deepfake Detection System

This project provides tools and scripts for preprocessing videos and training a deepfake detection model using the Xception architecture.

## Prerequisites

- Python 3.7 or higher
- ffmpeg installed and available in your system PATH

## Installation

To install the required Python dependencies, run:

```bash
pip install torch torchvision facenet-pytorch timm scikit-learn numpy Pillow
```

## Preprocessing

Before training, you need to preprocess your video dataset:

### 1. Extract frames from videos

Use `extract_frames.py` to extract frames from raw videos:

```bash
python deepfake-detection-system/preprocessing/extract_frames.py --video_path /path/to/video.mp4 --output_dir /path/to/frames --fps 1
```

### 2. Detect and align faces

Use `face_align_mtcnn.py` to detect and align faces in the extracted frames:

```bash
python deepfake-detection-system/preprocessing/face_align_mtcnn.py --frame_dir /path/to/frames --output_dir /path/to/aligned_faces
```

## Dataset Structure

Organize your processed dataset as follows for training:

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

## Training

Once your dataset is ready, start training with:

```bash
python deepfake-detection-system/train_xception_training.py
```

The script defaults to using `./dataset/` as dataset root and trains for 10 epochs. Model checkpoints are saved as `best_model.pth`.

## Notes

- The training script will automatically use GPU if available.
- Ensure `ffmpeg` is installed and accessible to use the frame extraction script.
- Adjust FPS and image size parameters as needed to fit your dataset requirements.

---

This README provides the full instructions to run preprocessing and training for the deepfake detection system.
