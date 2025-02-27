import cv2
import pytesseract
import numpy as np
import os
import requests

def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        print(f"Downloading model from {model_url} ...")
        r = requests.get(model_url, allow_redirects=True)
        with open(model_path, 'wb') as f:
            f.write(r.content)
        print("Download complete.")
    else:
        print("Model already exists.")

def ocr_urdu_tesseract_enhanced(image_path):
    # Read image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Image not found."

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply median blur to reduce noise
    denoised = cv2.medianBlur(gray, 3)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Define a kernel for morphological operations and apply them
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Resize image by a factor of 2
    scaled = cv2.resize(morph, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Apply histogram equalization
    equalized = cv2.equalizeHist(scaled)

    # Deskewing
    coords = np.column_stack(np.where(equalized > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = equalized.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(equalized, M, (w, h),
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Convert deskewed image (currently grayscale) to BGR
    deskewed_color = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)

    # Download and apply super-resolution
    model_url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x3.pb"
    model_path = "EDSR_x3.pb"
    download_model(model_url, model_path)

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel('edsr', 3)
    super_resolved = sr.upsample(deskewed_color)

    # Set Tesseract configuration for Urdu language
    config = r'--oem 3 --psm 6 -l urd'
    text = pytesseract.image_to_string(super_resolved, config=config)
    return text

# Example usage:
if __name__ == "__main__":
    image_path = "/content/fakeback.jpg"  # Replace with your image path
    urdu_text = ocr_urdu_tesseract_enhanced(image_path)
    print("OCR Output (Enhanced):")
    print(urdu_text)
