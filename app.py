# !pip install --no-cache-dir easyocr opencv-python pillow --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org
# !pip install easyocr opencv-python pillow Groq flask opencv-python easyocr pillow groq werkzeug
#only run:
# pip install --no-cache-dir easyocr opencv-python pillow groq flask werkzeug --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

import os
import cv2
import json
import re
import easyocr
import numpy as np
from flask import Flask, request, jsonify
from groq import Groq

# Define global placeholders for file paths.
# Update these with the actual paths when needed.
ORIGINAL_IMAGE_PATH = r"\idscans\uploadedid.jpg"   # Path where the original image will be saved
ENHANCED_IMAGE_PATH = r"\idscans\enhancedid.jpg"   # Path where the enhanced image will be saved
# CHANGE THE ABOVE PATHS AS PER REQ

# Initialize Flask app
app = Flask(__name__)

# Initialize Groq client (replace with your actual API key if needed)
groq_client = Groq(api_key="{YOUR_GROQ_API}")

def save_original_image(uploaded_file, save_path):
    """Save the uploaded file to the specified original image path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    uploaded_file.save(save_path)
    return save_path

def process_image(original_image_path, enhanced_image_path):
    """
    Processes the original image from the given path:
      1. Loads the image.
      2. Enhances it (converts to grayscale and applies thresholding).
      3. Saves the enhanced image to the specified enhanced image path.
      4. Uses EasyOCR to extract text from the enhanced image.
    """
    # Load the original image
    image = cv2.imread(original_image_path)
    if image is None:
        raise ValueError("Original image not found at the provided path.")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding (using OTSU for automated threshold selection)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Ensure the directory for the enhanced image exists
    os.makedirs(os.path.dirname(enhanced_image_path), exist_ok=True)
    cv2.imwrite(enhanced_image_path, thresh)
    
    # Extract text using EasyOCR
    reader = easyocr.Reader(['en'])
    extracted_text = reader.readtext(thresh, detail=0)
    
    print("Extracted Text:")
    print("\n".join(extracted_text))
    return extracted_text

def get_groq_completion(extracted_text):
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": (
                f"Given the following extracted text from a Pakistani ID card, extract and return a JSON object containing only "
                f"the following details: Name, Father Name, Gender, Country of Stay, Identity Number, Date of Birth, Date of Issue, "
                f"and Date of Expiry. If any field is missing, set its value to null. Do not include any extra text. "
                f"Data: {extracted_text} "
                f"Response: Please provide valid JSON only."
            )
        }],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False
    )
    return completion

def parse_json_response(response_content):
    print("Raw Response Content:")
    print(response_content)
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError as e:
                print("Error decoding extracted JSON:", e)
                return None
        print("No valid JSON found.")
        return None
    
@app.route("/")
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <title>ID OCR API</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
      <style>
        body { background-color: #f8f9fa; }
        .container { max-width: 600px; margin-top: 50px; }
      </style>
    </head>
    <body>
      <div class="container">
        <h1 class="mb-4 text-center">ID OCR API</h1>
        <p class="text-center">Upload your ID image and see the extracted details below.</p>
        <form id="uploadForm" enctype="multipart/form-data">
          <div class="mb-3">
            <input type="file" class="form-control" id="fileInput" name="file" required>
          </div>
          <div class="d-grid">
            <button type="submit" class="btn btn-primary">Process ID</button>
          </div>
        </form>
        <div id="result" class="mt-4"></div>
      </div>
      <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
          e.preventDefault();
          const fileInput = document.getElementById('fileInput');
          if(fileInput.files.length === 0) {
            alert("Please select a file.");
            return;
          }
          const formData = new FormData();
          formData.append('file', fileInput.files[0]);
          document.getElementById('result').innerHTML = '<p>Processing...</p>';
          try {
            const response = await fetch('/process-id', { method: 'POST', body: formData });
            const data = await response.json();
            if(data.status === "success") {
              document.getElementById('result').innerHTML = '<pre>' + JSON.stringify(data.extracted_data, null, 4) + '</pre>';
            } else {
              document.getElementById('result').innerHTML = '<p class="text-danger">Error: ' + data.message + '</p>';
            }
          } catch (error) {
            document.getElementById('result').innerHTML = '<p class="text-danger">Error: ' + error + '</p>';
          }
        });
      </script>
    </body>
    </html>
    """
    return html

@app.route("/process-id", methods=["POST"])
def process_id():
    """
    API endpoint that:
      1. Accepts an image file in the request body.
      2. Saves the original image to ORIGINAL_IMAGE_PATH.
      3. Processes the image to create an enhanced version saved at ENHANCED_IMAGE_PATH.
      4. Extracts text from the enhanced image.
      5. Sends the text to Groq for structured extraction and returns the JSON response.
    """
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part in request."}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file."}), 400

    try:
        # Save the original image to the specified path
        save_original_image(file, ORIGINAL_IMAGE_PATH)
        
        # Process the image and extract text from the enhanced version
        extracted_text = process_image(ORIGINAL_IMAGE_PATH, ENHANCED_IMAGE_PATH)
        
        # Get structured response from Groq
        completion = get_groq_completion(extracted_text)
        response_content = completion.choices[0].message.content.strip()
        print("Response Content:", response_content)
        
        response_json = parse_json_response(response_content)
        if response_json is None:
            return jsonify({"status": "error", "message": "Failed to parse valid JSON from response."}), 500
        
        return jsonify({"status": "success", "extracted_data": response_json})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
