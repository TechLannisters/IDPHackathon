import cv2 

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