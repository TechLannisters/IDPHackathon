import cv2 
image = cv2.imread('C:\\Users\\saini\\OneDrive\\Pictures\\ocr-test.png') 
import numpy as np
from PIL import Image 
import pytesseract
import re

# Preprocessing function
def preprocess_image(image_path, output_type='edges'):
    # 1. Load Image 
    image = cv2.imread(image_path)

    # 2. Grayscale Conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Noise Reduction
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10) 

    if output_type == 'edges':
        edges = cv2.Canny(denoised, 50, 150) 
        return edges
    elif output_type == 'thresholded':
        thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh
    else:
        return None  # Or raise an error if invalid output_type

def filter_and_group_lines(lines):
    horizontal_lines = []
    vertical_lines = []
    angle_tolerance = 10  # Degrees

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle) < angle_tolerance:
            horizontal_lines.append(line)
        elif abs(angle - 90) < angle_tolerance:
            vertical_lines.append(line)

    return horizontal_lines, vertical_lines

def estimate_cells(horizontal_lines, vertical_lines):
    #  Sort lines by coordinates for consistency
    horizontal_lines.sort(key=lambda x: x[0][1])
    vertical_lines.sort(key=lambda x: x[0][0])

    #  Simple (potential) cell estimation
    table_cells = []
    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            x1, y1 = vertical_lines[j][0][:2]
            x2, y2 = vertical_lines[j + 1][0][:2]
            y3, _ = horizontal_lines[i][0][:2]
            y4, _ = horizontal_lines[i + 1][0][:2]
            table_cells.append((x1, y3, x2 - x1, y4 - y3))

    return table_cells

def analyze_layout(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocessing 
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Hough Line Detection
    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    horizontal_lines, vertical_lines = filter_and_group_lines(lines)

    # Table Cell Estimation 
    table_cells = estimate_cells(horizontal_lines, vertical_lines) 

    # Find Text Blocks
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Analysis and Extraction
    extracted_data = {}  # A dictionary to store your results

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour) 

        if is_within_cell(table_cells, x, y, w, h):
            if y < 100:  # Assuming header is in the top portion
                header_text = extract_text_from_region(image, x, y, w, h)
                process_header(header_text) 
            elif y > image.shape[0] - 100:  # Assuming footer in the bottom
                footer_text = extract_text_from_region(image, x, y, w, h)
                process_footer(footer_text)
            else:
                # Potentially miscellaneous text
                print(f"Potential miscellaneous text at ({x}, {y})")  # Log it for now

        # Classify based on position and keywords
    if y < 100:  # Assuming header is in top portion
            header_text = extract_text_from_region(image, x, y, w, h)
            extracted_data['header'] = process_header(header_text)  
    elif is_within_cell(table_cells, x, y): 
            cell_text = extract_text_from_region(image, x, y, w, h)
            # Process based on expected item list layout 
    else:
            # ... Handle footer text or other regions
            return extracted_data 

def extract_text_from_region(image, x, y, w, h):
    roi = image[y:y+h, x:x+w]
    cell_text = extract_text_from_region(image, x, y, w, h)
    # Perform OCR on the ROI (Assuming you use Tesseract with PIL)
    pil_roi = Image.fromarray(roi)
    ocr_text = pytesseract.image_to_string(pil_roi)

    return ocr_text

def process_header(header_text):
    header_data = {}

    # Example: Using simple string manipulation and assumptions
    lines = header_text.splitlines()
    for line in lines:
        if "Invoice Number:" in line:
            header_data['invoice_number'] = line.split(':')[1].strip()
        elif "Date:" in line:
            header_data['date'] = line.split(':')[1].strip()
        # ... extract store info

    return header_data 

def is_within_cell(table_cells, x, y, w, h):
    # Find the center of the bounding box
    center_x, center_y = x + w // 2, y + h // 2

    for cell in table_cells:
        cell_x, cell_y, cell_w, cell_h = cell
        if cell_x <= center_x <= cell_x + cell_w and cell_y <= center_y <= cell_y + cell_h:
            return True 

    return False # Not found within any cell

def process_footer(footer_text):
    footer_data = {}  # A dictionary to store extracted information

    # Step 1: Split into Lines (if necessary)
    lines = footer_text.splitlines()

    return footer_data