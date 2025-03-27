import pyautogui
import cv2
import numpy as np
import datetime

def take_screenshot(region):
    screenshot = pyautogui.screenshot(region=region)
    return np.array(screenshot)


# Define the region of the screen to capture
# x1,y1,width,height=800,280,306,82 #for the captcha
# x1,y1,width,height=800,380,300,50 #for the begin test button
# x1,y1,width,height=1150,400,60,40 #for the verify button
x1,y1,width,height=900,390,300,50
region = (x1,y1,width,height)  # (x, y, width, height)


i = 0
start_time = datetime.datetime.now()
while i < 10:
    # Draw a red rectangle to visualize the region
    pyautogui.moveTo(x1, y1)
    pyautogui.dragTo(x1 + width, y1 + height, duration=1, button='left', tween=pyautogui.easeInOutQuad)
        
    i += 1
    # Capture a screenshot of the specified region
    screenshot = take_screenshot(region)

    #resize the image to fit the 250 50 size for captcha
    #screenshot = cv2.resize(screenshot, (250, 50) , interpolation = cv2.INTER_AREA) # for the captcha
    #screenshot = cv2.resize(screenshot, (300, 50) , interpolation = cv2.INTER_AREA) # for the begin test button
    #screenshot = cv2.resize(screenshot, (60, 40) , interpolation = cv2.INTER_AREA) # verify button
    screenshot = cv2.resize(screenshot, (300 ,50) , interpolation = cv2.INTER_AREA) 

    # Save the screenshot to disk
    cv2.imwrite(f"screenshots/screen_{i}.png", screenshot)



   


    
