# ID OCR Scanner & Face Extraction API

![image](https://github.com/user-attachments/assets/57628e0d-3712-4302-abda-e0709049d9a5)

This project is an AI-powered OCR application designed to extract information from Pakistani ID cards using computer vision and OCR techniques. The application performs two main tasks:

1. **OCR Extraction:**  
   Processes the uploaded ID image, enhances it (using OpenCV and EasyOCR), and extracts text. This text is then sent to a Groq API for structured extraction of key fields such as Name, Father Name, Gender, Country of Stay, Identity Number, Date of Birth, Date of Issue, and Date of Expiry.

2. **Face Extraction:**  
   Uses a YOLOv8 face detection model to detect and crop the face from the ID image. The cropped face is saved to a specified directory, and its file path is included in the API response.

> **Note:**  
> This project was developed as a contract assignment by a Junior AI Engineer from January to March [Year].  
> **IMPORTANT:** Update all placeholder paths and API keys as described below.

3. **Testing:**
   ```bash
   curl.exe -X POST -F "file=@{JPG_PATH}" http://127.0.0.1:5000/process-id

4. **UI+shell**
   works for both on command line and HTML webpage

5. **Directories & Models**
   The image directories paths must be carefully understood and setup

6. **FILES**
   - app.py is the main server file (complete on its own without dependence on other files)
   - id image folders contains the IDs
   - back_id.py is the backside ID work that is separate and unintegrated
   - faceID.py is the separate code of the face detection already integrated in app.py
   - serverless_code(raw) is the model and code that doesnt has server integrated

7. **Download YOLO model in the same directory**
   ```bash
   curl -L -o yolov8l-face-lindevs.pt https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8l-face-lindevs.pt
   
## Features

- **Image Upload & Processing:**  
  - Accepts image files via a POST request to `/process-id`.
  - Saves the original image to a user-specified path.
  - Enhances the image (grayscale conversion, thresholding, etc.) and saves the processed version to a separate path.
  
- **OCR Text Extraction:**  
  - Uses EasyOCR (configured for English and Urdu) to extract text from the enhanced image.
  - Sends the extracted text to a Groq API for structured information extraction.
  
- **Face Detection & Extraction:**  
  - Utilizes a YOLOv8 model (`yolov8l-face-lindevs.pt`) to detect and crop the face from the ID image.
  - Saves the cropped face image in a user-specified directory and returns its file path in the final response.
  
- **User-Friendly Interface:**  
  - A simple HTML homepage (with Bootstrap) is provided for testing the API via a web browser.

## Installation

### Dependencies

Install the required Python packages with this one-liner:

```bash
pip install --no-cache-dir easyocr opencv-python pillow groq ultralytics flask werkzeug --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
