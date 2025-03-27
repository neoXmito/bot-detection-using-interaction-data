import cv2
import pytesseract
import numpy as np

def extract_white_text(image_path):
    # Load image and convert to HSV
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create mask for reddish background (adjust these ranges)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Invert mask to isolate white text
    text_mask = cv2.bitwise_not(mask)
    
    # Enhance contrast using CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Combine masks and preprocessing
    processed = cv2.bitwise_and(enhanced, enhanced, mask=text_mask)
    _, thresh = cv2.threshold(processed, 200, 255, cv2.THRESH_BINARY)
    
    # OCR configuration
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    return text.strip()

# Usage
result = extract_white_text("screenshots\screen_2.png")
print("Extracted text:", result)