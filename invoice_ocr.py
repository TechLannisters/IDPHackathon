import cv2
import pytesseract
from PIL import Image
from preprocess import preprocess_image
import numpy as np
image = cv2.imread('C:\\Users\\saini\\OneDrive\\Pictures\\ocr-test.png') 

# Specify the path to your Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract\tesseract.exe'  

# --- Table Detection Functions---
def detect_table_lines(edges):
    rho = 1  
    theta = np.pi / 180  
    threshold = 100  
    min_line_length = 50  
    max_line_gap = 10  

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, min_line_length, max_line_gap)
    return lines 

def filter_and_group_lines(lines):
    horizontal_lines = []
    vertical_lines = []
    angle_tolerance = 10  # Adjust as needed

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle) < angle_tolerance:
            horizontal_lines.append(line)
        elif abs(angle - 90) < angle_tolerance:
            vertical_lines.append(line)

    return horizontal_lines, vertical_lines

def refine_and_draw_table(image, horizontal_lines, vertical_lines):
    # ... (Implement logic to refine line intersections and draw borders) ...
    # Example (Draws lines for visualization - modify as needed)
    for line in horizontal_lines + vertical_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image

# --- Main Invoice OCR Logic ---
def process_invoice(image_path):
    # ... (Loading Image and Preprocessing -  Same as before ) ...

    # Table Detection
    processed_image = preprocess_image(image.copy())
    lines = detect_table_lines(processed_image)
    horizontal_lines, vertical_lines = filter_and_group_lines(lines)
    image_with_table_borders = refine_and_draw_table(image.copy(), horizontal_lines, vertical_lines)

    # Apply OCR (with potential table-specific logic based on identified regions)
    preprocessed_image = preprocess_image(image) # Apply OCR-specific preprocessing
    pil_image = Image.fromarray(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)) 
    text = pytesseract.image_to_string(pil_image)
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT) 

# Specify your image path
image_path = 'C:\\Users\\saini\\OneDrive\\Pictures\\ocr-test.png' 

# Preprocess the image and get the result
preprocessed_image = preprocess_image(image_path) 

# --- Conversion to PIL ---
pil_image = Image.fromarray(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)) 

# --- Text Extraction with Bounding Boxes ---
# Get bounding box data using image_to_data
data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT) 


# Extract and visualize bounding boxes (optional)
for i, word in enumerate(data['text']):
  if word:
      x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
      cv2.rectangle(preprocessed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Example: Find potential invoice number regions
potential_invoice_regions = []
for i, word in enumerate(data['text']):
    if word.lower() == 'invoice':
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        potential_invoice_regions.append((x, y, w, h)) 

# (Further logic to select the most likely invoice number region will go here)

# Display the image with bounding boxes (optional)
cv2.imshow('Bounding Boxes', preprocessed_image)
cv2.waitKey(0)

# ---  Other Output Formats ---
# HOCR Output
hocr_data = pytesseract.image_to_string(preprocessed_image, output_type=pytesseract.Output.DICT + pytesseract.Output.HOCR)
with open('output.hocr', 'w') as f:
    f.write(hocr_data) 

# Searchable PDF Output
pdf_data = pytesseract.image_to_pdf_or_hocr(preprocessed_image, extension='pdf')
with open('output.pdf', 'wb') as f:
    f.write(pdf_data)

if __name__ == '__main__':
    image_path = 'C:\\Users\\saini\\OneDrive\\Pictures\\ocr-test.png' 
    process_invoice(image_path) 