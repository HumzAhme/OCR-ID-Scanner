# !apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-urd
# !pip install pytesseract pillow gradio arabic-reshaper python-bidi transformers numpy ultralytics
# !pip install layoutparser paddleocr opencv-python-headless torch easyocr torchvision pytorch
# !wget -O yolov8l-face-lindevs.pt https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8l-face-lindevs.pt


import cv2
import torch
import os
from ultralytics import YOLO
from google.colab.patches import cv2_imshow  # Display images in Colab

# Ensure save directory exists
save_dir = "/content/passpic"
os.makedirs(save_dir, exist_ok=True)

# Load YOLOv8 face detection model
model = YOLO("yolov8l-face-lindevs.pt")  # Use the downloaded model

# Function to detect and save the portrait
def extract_portrait_from_directory(directory_path):
    # Loop through all image files in the directory
    for filename in os.listdir(directory_path):
        # Process only image files (add more extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            print(f"Processing {image_path}...")

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image {filename}.")
                continue

            # Perform face detection
            results = model(image)

            # Process detections
            face_found = False
            for result in results:
                for box in result.boxes.xyxy:  # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box[:4])

                    # Crop and save the face
                    cropped_face = image[y1:y2, x1:x2]
                    face_filename = f"{os.path.splitext(filename)[0]}_face.jpg"
                    face_path = os.path.join(save_dir, face_filename)
                    cv2.imwrite(face_path, cropped_face)
                    print(f"✅ Face saved at: {face_path}")

                    # Show the cropped image in Colab
                    cv2_imshow(cropped_face)
                    face_found = True
                    break  # Stop once we find the first face

            if not face_found:
                print(f"⚠ No face detected in {filename}.")

# Example usage (replace '/content/images' with the path to your directory containing images)
directory_path = "/content/passID"  # Change this to your directory path
extract_portrait_from_directory(directory_path)
