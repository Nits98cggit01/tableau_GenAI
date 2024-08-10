import os
import pytesseract
import easyocr
import pandas as pd
from PIL import Image
import re

import keras_ocr
# import tensorflow as tf

ROOT_PATH = r"C:\\Users\\NITINS\\OneDrive - Capgemini\\CAPGEMINI\\PROJECT\\GEN AI\\report-usecase\\tableau\\tableau-auth\\pat\\890Portal\\Dashboard_images"
# img_name = 'CMO_Dashboard_Promotion Effect.png'
img_name = 'Peakon FS_Dashboard 2.png'

# # Load an image using PIL
image_path = os.path.join(ROOT_PATH,img_name)  # Replace with the path to your image
# image = Image.open(image_path)

# # Use pytesseract to extract text from the image
# extracted_text = pytesseract.image_to_string(image)

# # Print the extracted text
# print(extracted_text)

# Actual OCR Reader
# reader = easyocr.Reader(['en'],gpu=False)
# result = reader.readtext(image_path)
# check = pd.DataFrame(result,columns=['bbox','text','config'])
# # check.to_csv('OCR_Image.csv')
# print('Done')


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

def is_heading(bbox, text, min_font_size=18):
    # bbox is a list of coordinates: [(top_left), (top_right), (bottom_right), (bottom_left)]
    top_left, top_right, bottom_right, bottom_left = bbox
    height = bottom_right[1] - top_right[1]
    
    # Adjust the min_font_size according to your specific needs
    return height > min_font_size

# Load the image to get its dimensions
image = Image.open(image_path)
image_width, image_height = image.size

headings = [
    entry for entry in result
    if is_phrase(entry[1]) and is_heading(entry[0], entry[1])
]

headings_df = pd.DataFrame(headings, columns=['bbox', 'text', 'config'])
headings_text_df = headings_df[['text']]
print(headings_text_df)
headings_text_df.to_csv('OCRTest.csv')

# pipeline = keras_ocr.pipeline.Pipeline()
# result = pipeline.recognize([image_path])
# check = pd.DataFrame(result[0],columns=['text','bbox'])
# print(check)



