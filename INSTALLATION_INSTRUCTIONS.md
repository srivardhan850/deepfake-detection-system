# Installation Instructions for Deepfake Detection System Dependencies

This document provides instructions to install necessary dependencies to run the deepfake detection system successfully on Windows.

## 1. Install FFmpeg Optional

The project now uses OpenCV first for video frame extraction. Install FFmpeg only if your video format does not open correctly.

### Steps:

1. Download FFmpeg:

- Go to the official FFmpeg website: https://ffmpeg.org/download.html
- Under "Windows", click the link to a build by Gyan (https://www.gyan.dev/ffmpeg/builds/)
- Download the "Release" full build zip file (e.g., `ffmpeg-release-essentials.zip`)

2. Extract the ZIP file:

- Extract the ZIP file to a folder, e.g., `C:\ffmpeg`

3. Add FFmpeg to your system PATH:

- Open the Start Menu, search for "Environment Variables", and select "Edit the system environment variables"
- In the System Properties window, click the "Environment Variables..." button
- Under "System variables", find and select the "Path" variable, then click "Edit..."
- Click "New" and add the path to FFmpeg bin folder, e.g. `C:\ffmpeg\bin`
- Click OK on all dialogs to save

4. Verify installation:

- Open a new Command Prompt (cmd) and run:

  ```
  ffmpeg -version
  ```

- If it shows the FFmpeg version info, FFmpeg is correctly installed.

## 2. Install Required Python Packages

Some required Python libraries are missing. Install them using pip:

```bash
pip install -r requirements.txt
```

- `opencv-python` is required for face alignment.
- `torch` and related packages are required for training.
- `timm` is used for the Xception model.
- `facenet-pytorch` is required for face detection and alignment using MTCNN.

## 3. Add or Train the Model

The web app needs a checkpoint named `best_model.pth` in the project root.

If you already have a trained checkpoint, place it here:

```text
deepfake-detection-system/best_model.pth
```

If you need to train one, prepare a dataset with this structure:

```text
dataset/
  real/
    video_or_person_1/
      aligned_faces/
        image1.jpg
  fake/
    video_or_person_2/
      aligned_faces/
        image1.jpg
```

Then run:

```bash
python train_xception_training.py --dataset ./dataset --epochs 10 --batch-size 32 --output best_model.pth
```


## 4. Run the App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

After performing these steps, try running the system again from video upload to training. Please let me know if you encounter any further issues or need assistance.
