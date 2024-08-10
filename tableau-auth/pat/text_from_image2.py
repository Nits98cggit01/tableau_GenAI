import os
import easyocr
import pandas as pd
from PIL import Image
import re

import keras_ocr
# import tensorflow as tf

ROOT_PATH = r"C:\\Users\\NITINS\\OneDrive - Capgemini\\CAPGEMINI\\PROJECT\\GEN AI\\report-usecase\\tableau\\tableau-auth\\pat\\890Portal\\Dashboard_images"
# img_name = 'CMO_Dashboard_Promotion Effect.png'
img_name = 'Consolidated Cookbook_Comparison of Financial Metrics.png'
image_path = os.path.join(ROOT_PATH,img_name)

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Read text from the image
result = reader.readtext(image_path)

# Function to identify phrases
def is_phrase(text):
    if len(text) <= 1:
        return False
    
    if len(re.findall(r'\w+', text)) >= 1:
        # Check if the text starts with a capital letter
        return text[0].isupper()
    return False

# Calculate the font size for each text block
def get_font_size(bbox):
    top_left, top_right, bottom_right, bottom_left = bbox
    return bottom_right[1] - top_right[1]

# Identify the font sizes of all phrases
font_sizes = [(entry, get_font_size(entry[0])) for entry in result if is_phrase(entry[1])]
if not font_sizes:
    print("No valid phrases found")
    exit()

# Find the maximum font size in the image
max_font_size = max(font_sizes, key=lambda x: x[1])[1]

# Identify headings based on font size within a range around the max font size
def is_heading(bbox, font_size, max_font_size, tolerance=2):
    return max_font_size - tolerance <= font_size <= max_font_size + tolerance

def combine_wrapped_headings(headings, tolerance=10):
    combined_headings = []
    current_heading = ""
    previous_bottom = None
    
    for bbox, text, config, font_size in headings:
        top_left, top_right, bottom_right, bottom_left = bbox
        if previous_bottom and abs(top_left[1] - previous_bottom) <= tolerance:
            current_heading += " " + text
        else:
            if current_heading:
                combined_headings.append(current_heading)
            current_heading = text
        previous_bottom = bottom_left[1]

    if current_heading:
        combined_headings.append(current_heading)
    
    return combined_headings

# Filter headings based on dynamic font size and combine wrapped text
potential_headings = [
    (entry[0], entry[1], entry[2], font_size) for entry, font_size in font_sizes
    if is_heading(entry[0], font_size, max_font_size)
]

# Combine wrapped headings
combined_headings = combine_wrapped_headings(potential_headings)

# Convert to DataFrame and select only the 'text' column
headings_df = pd.DataFrame(combined_headings, columns=['text'])

# Save to CSV (if needed)
headings_df.to_csv('OCRTest.csv', index=False)

# Print the DataFrame with only the 'text' column
print(headings_df)

print('Done')
