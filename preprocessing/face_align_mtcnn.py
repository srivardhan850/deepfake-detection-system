import os
import cv2
import torch
from facenet_pytorch import MTCNN
import argparse
from PIL import Image
from torchvision.transforms.functional import to_pil_image

def detect_and_align_faces(frame_dir, output_dir, image_size=224, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Detect and align faces from extracted frames using MTCNN.

    Args:
        frame_dir (str): Directory with extracted frames.
        output_dir (str): Directory to save aligned face crops.
        image_size (int): Size of output crops (square).
        device (str): Device to run MTCNN on.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize MTCNN
    mtcnn = MTCNN(image_size=image_size, margin=20, device=device, keep_all=True)

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Processing {len(frame_files)} frames for face detection.")

    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        img = Image.open(frame_path).convert('RGB')

        # Detect faces and crop aligned
        boxes, probs = mtcnn.detect(img)

        if boxes is None:
            print(f"No faces detected in {frame_file}")
            continue

        for i, box in enumerate(boxes):
            # Crop and save each detected face
            face_aligned = mtcnn.extract(img, [box], save_path=None)[0]
            if torch.is_tensor(face_aligned):
                face_img = to_pil_image(face_aligned.cpu()).convert('RGB')
            else:
                face_img = face_aligned.convert('RGB')
            face_img = face_img.resize((image_size, image_size))
            save_path = os.path.join(output_dir, f"{os.path.splitext(frame_file)[0]}_face{i+1}.jpg")
            face_img.save(save_path)

    print(f"Face detection and alignment completed. Saved crops to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detect and align with MTCNN")
    parser.add_argument('--frame_dir', type=str, required=True, help='Directory with extracted frames')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save aligned faces')
    parser.add_argument('--image_size', type=int, default=224, help='Size of output face crops')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detect_and_align_faces(args.frame_dir, args.output_dir, args.image_size, device)
