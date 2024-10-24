#import libraries
import cv2
import os
import numpy as np
from scipy import ndimage

# Loading image
IMAGE_PATH = r'C:\Users\Arjun Koirala\pythonko\WhatsApp Image 2024-10-21 at 22.25.07_07dd020a.jpg'  
image = cv2.imread(IMAGE_PATH)

# Grayscale Conversion & Blurring
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Binarization by Otsu's Thresholding
thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Morphological Operations to Enhance Character Separation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
eroded = cv2.erode(thresh, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel, iterations=1)

# Find Contours to get Potential Characters
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter Contours by Area and Aspect Ratio to reduce Noise
min_area = 100
max_area = 1000
min_aspect_ratio = 0.1
max_aspect_ratio = 5

filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h
    if min_area < area < max_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
        filtered_contours.append(contour)

# Draw Bounding Rectangles around Filtered Contours (Characters)
character_image = image.copy()
cv2.drawContours(character_image, filtered_contours, -1, (0, 255, 0), 1)

# Save Each Character on corresponding folder
CHARACTER_FOLDER = 'devanagari_characters/'  
os.makedirs(CHARACTER_FOLDER, exist_ok=True)

for i, contour in enumerate(filtered_contours):
    
    x, y, w, h = cv2.boundingRect(contour)
    roi = image[y:y+h, x:x+w]
    cv2.imwrite(CHARACTER_FOLDER + f'character_{i+1}.png', roi)
