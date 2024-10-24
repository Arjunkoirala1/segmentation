import cv2
import numpy as np
import os

# Load the image
IMAGE_PATH = r'C:\Users\Arjun Koirala\pythonko\devnagari.png'
image = cv2.imread(IMAGE_PATH)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not open or find the image.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Dilation to connect text components
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=2)  # Increase iterations if necessary

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a list to store line data
lines = []

# Iterate through contours to create bounding boxes for lines
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    # Filter based on height to consider only horizontal lines
    if h > 10:  # Adjust minimum height for a line if necessary
        lines.append((y, x, w, h))  # Store line information

# Sort lines by their y-coordinate (top to bottom)
lines = sorted(lines, key=lambda line: line[0])

# Create an output folder for words
OUTPUT_FOLDER = 'extracted_words'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize a word counter for naming files
word_counter = 1

# Iterate through the detected lines
for line_index, (y, x, w, h) in enumerate(lines):
    # Create a region of interest (ROI) for each line
    line_roi = dilated[y:y + h, x:x + w]

    # Find words in the line ROI
    line_contours, _ = cv2.findContours(line_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through word contours
    for wc in line_contours:
        wx, wy, ww, wh = cv2.boundingRect(wc)
        if ww > 5:  # Minimum width for a word
            # Extract the word from the original image
            word_roi = image[y:y + h, x + wx:x + wx + ww]
            if word_roi.size > 0:  # Check if the extracted ROI is valid
                # Save the word image
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, f'word_{word_counter}.png'), word_roi)
                word_counter += 1

print(f"Extracted {word_counter - 1} words into '{OUTPUT_FOLDER}' folder.")
