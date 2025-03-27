import cv2
import random
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt


# Load models
model = tf.keras.models.load_model("models/FINALMODELS/captchasolve.h5")
label_encoder = joblib.load("models/trained_label_encoder.pkl")
model_test = YOLO("models/yolo_model2/best.pt")
print("Models loaded successfully")

def show_image(image):
    """Display an image using matplotlib"""
    plt.close()
    plt.imshow(image, cmap="gray")
    plt.show()


def check_letter_tarakom(pic):
    """Check if a letter is well-separated or too compacted"""
    _, pic = cv2.threshold(pic, 127, 255, cv2.THRESH_BINARY)
    s = 90 - (np.sum(pic, axis=0, keepdims=True) / 255)
    total = len(s[0])
    howmanyblack = 0
    for i in s[0]:
        if np.sum(i) >= 175:
            howmanyblack += 1
    return total - howmanyblack <= 22


def get_allpredicted(photo):
    """
    Process an input photo (as a NumPy array) with the YOLO model to extract letters,
    then uses the Keras model to predict each letter and returns the predicted vector.

    Parameters:
        photo (numpy.ndarray): The input image in BGR format.

    Returns:
        list: A list of predicted characters (the predicted vector).
    """
    # Run YOLO detection on the photo
    results = model_test(photo)
    detections = results[0].boxes
    # Sort detections by the x-coordinate (left to right)
    detections = sorted(detections, key=lambda box: box.xyxy[0][0])
    
    letters = []
    last_box = []
    howmany = 0

    # Loop over each detected bounding box
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Ensure a new box is far enough from the previous one (to avoid duplicates)
        if last_box:
            if x1 - last_box[0] > 10:
                letter_crop = photo[y1:y2, x1:x2]
                resized_letter = cv2.resize(letter_crop, (32, 52))
                resized_letter = cv2.cvtColor(resized_letter, cv2.COLOR_BGR2GRAY)
                if howmany == 8:
                    if check_letter_tarakom(letter_crop):
                        _, _ = cv2.threshold(resized_letter, 128, 255, cv2.THRESH_BINARY)
                        letters.append(resized_letter)
                elif howmany <= 7:
                    _, _ = cv2.threshold(resized_letter, 128, 255, cv2.THRESH_BINARY)
                    letters.append(resized_letter)
                last_box = [x1, y1, x2, y2]
                howmany += 1
        else:
            letter_crop = photo[y1:y2, x1:x2]
            resized_letter = cv2.resize(letter_crop, (32, 52))
            resized_letter = cv2.cvtColor(resized_letter, cv2.COLOR_BGR2GRAY)
            if check_letter_tarakom(letter_crop):
                _, _ = cv2.threshold(resized_letter, 128, 255, cv2.THRESH_BINARY)
                letters.append(resized_letter)
            last_box = [x1, y1, x2, y2]
            howmany += 1

    print(f"Number of letters extracted: {len(letters)}")
    
    allpredicted = []
    # Predict each letter using the TensorFlow model
    for letter in letters:
        sample_image = np.array(letter)
        sample_image = sample_image.reshape(1, 52, 32, 1)
        predicted_label = model.predict(sample_image)
        predicted_class = label_encoder.inverse_transform([predicted_label.argmax()])
        allpredicted.append(predicted_class[0])
    
    return allpredicted

'''
# Test the function with a random image from the dataset
rendnum = random.choice(range(1, 10))
image_path_test = f"screenshots/screen_{rendnum}.png"
print(f"Processing image number: {rendnum}")
image_test = cv2.imread(image_path_test)

# Run the function
predicted_vector = get_allpredicted(image_test)

# Print the result
captcha_result = "".join(predicted_vector)
print("Predicted CAPTCHA:", captcha_result)

'''

#make a loop to automatically pass the human verification--------------------------------------
import cv2
import pytesseract
import numpy as np
import time
import pyautogui

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

def take_screenshot(region):
    screenshot = pyautogui.screenshot(region=region)
    return np.array(screenshot)

def photo_begin_test():
    region = (800, 380, 300, 50)
    screenshot = take_screenshot(region)
    screenshot = cv2.resize(screenshot, (300, 50) , interpolation = cv2.INTER_AREA) # for the begin test button
    cv2.imwrite(f"screenshots/begin_test.png", screenshot)
    if extract_white_text("screenshots/begin_test.png") == "BeginTest":
        return True

def photo_verify():
    region = (1150, 400, 60, 40)
    screenshot = take_screenshot(region)
    screenshot = cv2.resize(screenshot, (60, 40) , interpolation = cv2.INTER_AREA) # verify button
    cv2.imwrite(f"screenshots/verify.png", screenshot)
    if extract_white_text("screenshots/verify.png") == "Verify":
        return True

nb_tries= 3
i=0
# Loop over the number of tries
while i < nb_tries:
    
    if photo_begin_test():
        #drag mouse to the begin test button
        pyautogui.dragTo(900, 390, duration=0.5, button='left', tween=pyautogui.easeInOutQuad)
        #click the begin test button
        pyautogui.click(x=900, y=390)
        time.sleep(0.5)
        #Drawg mouse to text input
        pyautogui.dragTo(900 + 50, 390 + 30, duration=0.5, button='left', tween=pyautogui.easeInOutQuad)
        #click the text input
        pyautogui.click(x=900 + 50, y=390 + 30)
        # press ctrl + A to select all
        pyautogui.hotkey('ctrl', 'a')
        
        #extract the captcha
        screenshot=take_screenshot((800, 280, 306, 82))
        screenshot = cv2.resize(screenshot, (250, 50) , interpolation = cv2.INTER_AREA)
        cv2.imwrite(f"screenshots/captcha.png", screenshot)
        print(f"Processing image number: {i}")
        image_test = cv2.imread("screenshots/captcha.png")
        # Run the function
        predicted_vector = get_allpredicted(image_test)

        #type the letter one by one from the vector
        for letter in predicted_vector:
            pyautogui.typewrite(letter)    
            time.sleep(random.uniform(0.1, 2))
        time.sleep(0.5)

        #click the verify button
        pyautogui.dragTo(1170, 420, duration=0.5, button='left', tween=pyautogui.easeInOutQuad)
        pyautogui.click(x=1170, y=420)
        i+=1

    time.sleep(1)

                           
