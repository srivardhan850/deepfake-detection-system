import os
import subprocess
import threading
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash, jsonify

# Additional imports for inference and face detection
from facenet_pytorch import MTCNN
from PIL import Image
import torch
from inference import get_model, predict_image

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Paths configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
FRAME_FOLDER = os.path.join(BASE_DIR, 'processed_frames')
ALIGNED_FOLDER = os.path.join(BASE_DIR, 'aligned_faces')
DATASET_FOLDER = os.path.join(BASE_DIR, 'dataset')

# Create directories if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(ALIGNED_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Global variable to hold processing logs
processing_logs = []

# Initialize MTCNN face detector and model globally to reuse per request
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=224, margin=20, device=device, keep_all=True)
model, model_device = get_model(device=device)

def run_command(command, description):
    """Run a shell command and capture output line by line for logs."""
    processing_logs.append(f"Starting: {description}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    for line in process.stdout:
        processing_logs.append(line.strip())
    process.wait()
    if process.returncode == 0:
        processing_logs.append(f"Completed: {description}")
    else:
        processing_logs.append(f"Failed: {description} with return code {process.returncode}")

def clear_folders():
    """Clear previous outputs before new run."""
    import shutil
    for folder in [FRAME_FOLDER, ALIGNED_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

def process_video(video_path):
    """Run the preprocessing and training pipeline on the uploaded video."""
    global processing_logs
    processing_logs = []
    clear_folders()

    # Step 1: Extract frames
    command_extract = [
        'python', os.path.join(BASE_DIR, 'preprocessing', 'extract_frames.py'),
        '--video_path', video_path,
        '--output_dir', FRAME_FOLDER,
        '--fps', '1'
    ]
    run_command(command_extract, "Extracting frames")

    # Step 2: Face detection and alignment
    command_align = [
        'python', os.path.join(BASE_DIR, 'preprocessing', 'face_align_mtcnn.py'),
        '--frame_dir', FRAME_FOLDER,
        '--output_dir', ALIGNED_FOLDER
    ]
    run_command(command_align, "Face detection and alignment")

    # Optional: Organize dataset structure for training
    import shutil
    real_dir = os.path.join(DATASET_FOLDER, 'real', 'video1', 'aligned_faces')
    if os.path.exists(real_dir):
        shutil.rmtree(real_dir)
    os.makedirs(real_dir, exist_ok=True)
    # Move aligned faces to dataset structure
    for f in os.listdir(ALIGNED_FOLDER):
        shutil.copy(os.path.join(ALIGNED_FOLDER, f), real_dir)

    # Step 3: Run training
    command_train = [
        'python', os.path.join(BASE_DIR, 'train_xception_training.py')
    ]
    run_command(command_train, "Training model")

def process_image(image_path):
    """Process uploaded image for deepfake detection."""
    global processing_logs
    processing_logs = []
    processing_logs.append(f"Processing image: {image_path}")

    # Load image with PIL
    img = Image.open(image_path).convert('RGB')

    # Detect faces in image
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        processing_logs.append("No faces detected in the image.")
        processing_logs.append("Image processing completed. Result: Not AI Generated")
        return

    # Extract aligned faces (MTCNN extract automatically crops aligned faces)
    faces = mtcnn.extract(img, boxes, save_path=None)

    results_summary = []
    for i, face in enumerate(faces):
        # face is a PIL Image of size 224x224 aligned
        pred_label, confidence = predict_image(model, model_device, face)
        result_str = f"Face {i+1}: {pred_label} (confidence: {confidence:.3f})"
        processing_logs.append(result_str)
        results_summary.append(pred_label)

    if any(label == 'fake' for label in results_summary):
        final_result = "AI Generated"
    else:
        final_result = "Not AI Generated"

    processing_logs.append(f"Image processing completed. Result: {final_result}")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    file_video = request.files.get('video')
    file_image = request.files.get('image')

    if (file_video is None or file_video.filename == '') and (file_image is None or file_image.filename == ''):
        flash('No file part')
        return redirect(url_for('index'))

    if file_video and file_video.filename != '':
        filename = file_video.filename
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file_video.save(video_path)

        # Run processing in a separate thread to avoid blocking
        thread = threading.Thread(target=process_video, args=(video_path,))
        thread.start()

        flash('Video uploaded successfully and processing started.')
        return redirect(url_for('status'))

    elif file_image and file_image.filename != '':
        filename = file_image.filename
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file_image.save(image_path)

        # Run image processing in separate thread
        thread = threading.Thread(target=process_image, args=(image_path,))
        thread.start()

        flash('Image uploaded successfully and processing started.')
        return redirect(url_for('status'))

    else:
        flash('No selected file')
        return redirect(url_for('index'))

@app.route('/status', methods=['GET'])
def status():
    return render_template('status.html', logs=processing_logs)

@app.route('/logs', methods=['GET'])
def logs():
    return jsonify(processing_logs)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
