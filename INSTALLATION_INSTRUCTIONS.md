# Installation Instructions for Deepfake Detection System Dependencies

This document provides instructions to install necessary dependencies to run the deepfake detection system successfully on Windows.

## 1. Install FFmpeg

FFmpeg is required for extract_frames.py to extract frames from videos.

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
pip install opencv-python torch torchvision timm facenet-pytorch
```

- `opencv-python` is required for face alignment.
- `torch` and related packages are required for training.
- `timm` is used for the Xception model.
- `facenet-pytorch` is required for face detection and alignment using MTCNN.


## 3. Improve Training Code to Handle Empty Dataset (Optional)

The training script currently crashes if there is no data. You may want to update `train_xception_training.py` to add a check before training starts to ensure dataset is not empty.

You can add at the start of `main()` or training function:

```python
if len(train_loader.dataset) == 0:
    print("Training dataset is empty. Please check preprocessing steps.")
    return
```

---

After performing these steps, try running the system again from video upload to training. Please let me know if you encounter any further issues or need assistance.
