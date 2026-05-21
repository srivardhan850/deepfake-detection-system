# Deepfake Detection System

This project provides a Flask web app plus training and preprocessing scripts for deepfake detection using an Xception image classifier on aligned face crops.

## Prerequisites

- Python 3.10 recommended
- FFmpeg optional. Video frame extraction uses OpenCV first and falls back to FFmpeg if needed.
- A trained model checkpoint named `best_model.pth` in the project root

## Installation

To install the required Python dependencies, run:

```bash
pip install -r requirements.txt
```

## Run the Web App

After installing dependencies and placing `best_model.pth` in the project root:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The app accepts images and videos. Videos are processed by extracting frames, detecting faces, predicting each detected face, and averaging the fake probability. The app does not train a model during upload.

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
python train_xception_training.py --dataset ./dataset --epochs 10 --batch-size 32 --output best_model.pth
```

The script defaults to using `./dataset/` as dataset root and saves the best checkpoint as `best_model.pth`.

## Notes

- The training script will automatically use GPU if available.
- Install FFmpeg only if OpenCV cannot read your video format.
- Adjust FPS and image size parameters as needed to fit your dataset requirements.
- Accuracy depends on the dataset. A model trained on a tiny or one-class dataset will not be reliable.

---

This README provides the full instructions to run preprocessing and training for the deepfake detection system.
