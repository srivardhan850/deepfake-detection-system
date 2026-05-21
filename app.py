import os
import subprocess
import threading
from statistics import mean
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

# Additional imports for inference and face detection
from facenet_pytorch import MTCNN
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
from inference import get_model, predict_image

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-me')

# Paths configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
FRAME_FOLDER = os.path.join(BASE_DIR, 'processed_frames')
ALIGNED_FOLDER = os.path.join(BASE_DIR, 'aligned_faces')
DATASET_FOLDER = os.path.join(BASE_DIR, 'dataset')
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pth')
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
FAKE_THRESHOLD = float(os.environ.get('FAKE_THRESHOLD', '0.45'))

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
model = None
model_device = device
model_error = None

try:
    model, model_device = get_model(model_path=MODEL_PATH, device=device)
except Exception as exc:
    model_error = str(exc)


def allowed_file(filename, allowed_extensions):
    _, ext = os.path.splitext(filename.lower())
    return ext in allowed_extensions


def require_model():
    if model is None:
        message = model_error or "Model is not loaded."
        processing_logs.append(f"Model unavailable: {message}")
        processing_logs.append("Result: Cannot run detection until best_model.pth is available.")
        return False
    return True


def face_to_pil(face):
    if isinstance(face, Image.Image):
        return face.convert('RGB')
    if torch.is_tensor(face):
        return to_pil_image(face.cpu()).convert('RGB')
    raise TypeError(f"Unsupported face crop type: {type(face)}")

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
    """Run video deepfake inference on the uploaded video."""
    global processing_logs
    processing_logs = []
    clear_folders()

    if not require_model():
        return

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

    face_files = sorted(
        f for f in os.listdir(ALIGNED_FOLDER)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )
    if not face_files:
        processing_logs.append("No faces were found in the extracted video frames.")
        processing_logs.append("Video processing completed. Result: No Face Detected")
        return

    fake_scores = []
    for face_file in face_files:
        face_path = os.path.join(ALIGNED_FOLDER, face_file)
        img = Image.open(face_path).convert('RGB')
        pred_label, confidence, fake_probability = predict_image(model, model_device, img)
        fake_scores.append(fake_probability)
        processing_logs.append(
            f"{face_file}: {pred_label} "
            f"(confidence: {confidence:.3f}, fake probability: {fake_probability:.3f})"
        )

    avg_fake_probability = mean(fake_scores)
    max_fake_probability = max(fake_scores)
    final_result = (
        "AI Generated / Deepfake"
        if avg_fake_probability >= FAKE_THRESHOLD or max_fake_probability >= 0.65
        else "Not AI Generated / Real"
    )
    processing_logs.append(f"Analysed faces: {len(face_files)}")
    processing_logs.append(f"Average fake probability: {avg_fake_probability:.3f}")
    processing_logs.append(f"Highest frame fake probability: {max_fake_probability:.3f}")
    processing_logs.append(f"Decision threshold: {FAKE_THRESHOLD:.2f}")
    processing_logs.append(f"Video processing completed. Result: {final_result}")

def process_image(image_path):
    """Process uploaded image for deepfake detection."""
    global processing_logs
    processing_logs = []
    processing_logs.append(f"Processing image: {image_path}")

    if not require_model():
        return

    # Load image with PIL
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as exc:
        processing_logs.append(f"Could not open image: {exc}")
        processing_logs.append("Image processing completed. Result: Invalid Image")
        return

    # Detect faces in image
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        processing_logs.append("No faces detected in the image.")
        pred_label, confidence, fake_probability = predict_image(model, model_device, img)
        final_result = "AI Generated / Deepfake" if fake_probability >= FAKE_THRESHOLD else "Not AI Generated / Real"
        processing_logs.append(
            f"Full image: {pred_label} "
            f"(confidence: {confidence:.3f}, fake probability: {fake_probability:.3f})"
        )
        processing_logs.append(f"Decision threshold: {FAKE_THRESHOLD:.2f}")
        processing_logs.append(
            "No-face fallback used. Result may be less reliable because the model was trained mostly on faces."
        )
        processing_logs.append(f"Image processing completed. Result: {final_result}")
        return

    # Extract aligned faces (MTCNN extract automatically crops aligned faces)
    faces = mtcnn.extract(img, boxes, save_path=None)

    results_summary = []
    fake_scores = []
    for i, face in enumerate(faces):
        face_img = face_to_pil(face)
        pred_label, confidence, fake_probability = predict_image(model, model_device, face_img)
        result_str = (
            f"Face {i+1}: {pred_label} "
            f"(confidence: {confidence:.3f}, fake probability: {fake_probability:.3f})"
        )
        processing_logs.append(result_str)
        results_summary.append(pred_label)
        fake_scores.append(fake_probability)

    avg_fake_probability = mean(fake_scores)
    max_fake_probability = max(fake_scores)
    final_result = "AI Generated / Deepfake" if max_fake_probability >= FAKE_THRESHOLD else "Not AI Generated / Real"

    processing_logs.append(f"Average fake probability: {avg_fake_probability:.3f}")
    processing_logs.append(f"Highest face fake probability: {max_fake_probability:.3f}")
    processing_logs.append(f"Decision threshold: {FAKE_THRESHOLD:.2f}")
    processing_logs.append(f"Image processing completed. Result: {final_result}")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', model_loaded=model is not None, model_error=model_error)

@app.route('/upload', methods=['POST'])
def upload_video():
    file_video = request.files.get('video')
    file_image = request.files.get('image')

    if (file_video is None or file_video.filename == '') and (file_image is None or file_image.filename == ''):
        flash('Please choose an image or video file before clicking Upload & Process.')
        return redirect(url_for('index'))

    if file_video and file_video.filename != '':
        if not allowed_file(file_video.filename, ALLOWED_VIDEO_EXTENSIONS):
            flash('Unsupported video type. Please upload mp4, mov, avi, mkv, or webm.')
            return redirect(url_for('index'))

        filename = secure_filename(file_video.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file_video.save(video_path)

        # Run processing in a separate thread to avoid blocking
        thread = threading.Thread(target=process_video, args=(video_path,))
        thread.start()

        flash('Video uploaded successfully and processing started.')
        return redirect(url_for('status'))

    elif file_image and file_image.filename != '':
        if not allowed_file(file_image.filename, ALLOWED_IMAGE_EXTENSIONS):
            flash('Unsupported image type. Please upload jpg, jpeg, png, or webp.')
            return redirect(url_for('index'))

        filename = secure_filename(file_image.filename)
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
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
