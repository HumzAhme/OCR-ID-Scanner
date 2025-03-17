from flask import Flask, request, jsonify
import os
import cv2
import json
import re
import easyocr
import numpy as np
from werkzeug.utils import secure_filename
from groq import Groq
import traceback

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Groq client
groq_api_key = os.getenv('GROQ_API_KEY')
groq_client = Groq(api_key="{HASSAN_YOUR_KEY}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Extract text using EasyOCR
        reader = easyocr.Reader(['en'])
        extracted_text = reader.readtext(thresh, detail=0)
        
        if not extracted_text:
            raise ValueError("No text extracted from image")
        
        return extracted_text
    except Exception as e:
        raise Exception(f"Image processing failed: {str(e)}")

def extract_dates(text_list):
    date_pattern = r'\d{2}[.,]\d{2}[.,]\d{4}'
    dates = []
    
    for text in text_list:
        matches = re.findall(date_pattern, text)
        dates.extend(matches)
    
    date_dict = {
        "Date of Birth": dates[0] if len(dates) > 0 else None,
        "Date of Issue": dates[1] if len(dates) > 1 else None,
        "Date of Expiry": dates[2] if len(dates) > 2 else None
    }
    
    return date_dict

def get_groq_completion(extracted_text, dates):
    try:
        prompt = (
            f"Given the following extracted text from a Pakistani ID card and the correct dates mapping: {dates}, "
            f"extract and return a JSON object containing only the following details: "
            f"Name, Father Name, Gender, Country of Stay, Identity Number, Date of Birth, Date of Issue, and Date of Expiry. "
            f"Use the dates provided in the dates mapping. "
            f"If any fields are not present in the data, set their value to null. "
            f"Data: {extracted_text}"
        )
        
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=False
        )
        return completion
    except Exception as e:
        raise Exception(f"Groq API call failed: {str(e)}")

def parse_json_response(response_content):
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r'(?<![\{\s,])"(?=[^"]*":)', '"', json_str)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                raise Exception(f"JSON parsing failed: {str(e)}")
        raise Exception("No valid JSON found in response")

def validate_and_format_dates(response_json, extracted_dates):
    if response_json and isinstance(response_json, dict):
        response_json['Date of Birth'] = extracted_dates['Date of Birth']
        response_json['Date of Issue'] = extracted_dates['Date of Issue']
        response_json['Date of Expiry'] = extracted_dates['Date of Expiry']
    return response_json

@app.route('/extract-id-data', methods=['POST'])
def extract_id_data():
    try:
        # Check if image file is present in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if a file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400
        
        try:
            # Save file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Process the image
            extracted_text = process_image(filepath)
            
            # Extract dates
            extracted_dates = extract_dates(extracted_text)
            
            # Get completion from Groq
            completion = get_groq_completion(extracted_text, extracted_dates)
            response_content = completion.choices[0].message.content.strip()
            
            # Parse and validate JSON response
            response_json = parse_json_response(response_content)
            response_json = validate_and_format_dates(response_json, extracted_dates)
            
            if not response_json:
                raise Exception("Failed to generate valid JSON response")
            
            return jsonify(response_json), 200
            
        except Exception as e:
            return jsonify({'error': f'Data extraction failed: {str(e)}'}), 500
            
        finally:
            # Clean up: delete temporary file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
