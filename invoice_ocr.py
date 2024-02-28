import cv2
import pytesseract
from PIL import Image

# Specify the path to your Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract\tesseract.exe'  

# Preprocessing function
def preprocess_image(image_path):
    # 1. Load Image 
    image = cv2.imread(image_path)

    # 2. Grayscale Conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Noise Reduction
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10)  

    # 4. Thresholding
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return thresh

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
