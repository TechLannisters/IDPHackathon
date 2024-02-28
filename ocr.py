from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract\tesseract.exe'  
image_path = 'C:\\Users\\saini\\OneDrive\\Pictures\\ocr-test.png' 
text = pytesseract.image_to_string(Image.open(image_path))
print(text)