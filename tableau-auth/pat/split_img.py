from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def split_dashboard_image1(image_path, output_dir):
    # Load the image using PIL
    image = Image.open(image_path)

    # Convert the image to an OpenCV image
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Use thresholding to create a binary image
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Dilate the image to close gaps in contours
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around each visual
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Filter out small boxes that might not be actual visuals
    min_area = 10000  # Adjust this threshold based on the visuals size
    filtered_boxes = [box for box in bounding_boxes if box[2] * box[3] > min_area]

    # Sort the boxes from top to bottom, then left to right
    filtered_boxes = sorted(filtered_boxes, key=lambda x: (x[1], x[0]))

    # Draw the bounding boxes on the original image for visualization
    for (x, y, w, h) in filtered_boxes:
        cv_image = cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert back to PIL image for display
    output_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    # Display the image with bounding boxes
    plt.imshow(output_image)
    plt.axis('on')
    plt.show()

    # Save individual visuals
    visuals = []
    for i, (x, y, w, h) in enumerate(filtered_boxes):
        visual = image.crop((x, y, x + w, y + h))
        visuals.append(visual)
        visual.save(f"{output_dir}/visual_{i+1}.png")

    # List the saved visuals
    visual_files = [f"{output_dir}/visual_{i+1}.png" for i in range(len(filtered_boxes))]
    return visual_files

def split_dashboard_image2(image_path, output_dir):
    # Load the image using PIL
    image = Image.open(image_path)

    # Convert the image to an OpenCV image
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Use thresholding to create a binary image
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Dilate the image to close gaps in contours
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around each visual
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Filter out small boxes that might not be actual visuals
    min_area = 10000  # Adjust this threshold based on the visuals size
    filtered_boxes = [box for box in bounding_boxes if box[2] * box[3] > min_area]

    # Expand the bounding boxes to include titles and legends
    expanded_boxes = []
    for (x, y, w, h) in filtered_boxes:
        padding = 10  # Adjust padding as needed
        new_x = max(x - padding, 0)
        new_y = max(y - padding, 0)
        new_w = min(w + 2 * padding, cv_image.shape[1] - new_x)
        new_h = min(h + 2 * padding, cv_image.shape[0] - new_y)
        expanded_boxes.append((new_x, new_y, new_w, new_h))

    # Sort the boxes from top to bottom, then left to right
    expanded_boxes = sorted(expanded_boxes, key=lambda x: (x[1], x[0]))

    # Draw the bounding boxes on the original image for visualization
    for (x, y, w, h) in expanded_boxes:
        cv_image = cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert back to PIL image for display
    output_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    # Display the image with bounding boxes
    plt.imshow(output_image)
    plt.axis('on')
    plt.show()

    # Save individual visuals
    visuals = []
    for i, (x, y, w, h) in enumerate(expanded_boxes):
        visual = image.crop((x, y, x + w, y + h))
        visuals.append(visual)
        visual.save(f"{output_dir}/visual_{i+1}.png")

    # List the saved visuals
    visual_files = [f"{output_dir}/visual_{i+1}.png" for i in range(len(expanded_boxes))]
    return visual_files

def split_dashboard_image3(image_path, output_dir):
    # Load the image using PIL
    image = Image.open(image_path)

    # Convert the image to an OpenCV image
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Use thresholding to create a binary image
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Dilate the image to close gaps in contours
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around each visual
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Filter out small boxes that might not be actual visuals
    min_area = 10000  # Adjust this threshold based on the visuals size
    filtered_boxes = [box for box in bounding_boxes if box[2] * box[3] > min_area]

    # Expand the bounding boxes to include titles and legends
    expanded_boxes = []
    for (x, y, w, h) in filtered_boxes:
        padding = 10  # Adjust padding as needed
        title_height = 30  # Adjust title height based on the title size
        new_x = max(x - padding, 0)
        new_y = max(y - padding - title_height, 0)  # Include space above for titles
        new_w = min(w + 2 * padding, cv_image.shape[1] - new_x)
        new_h = min(h + 2 * padding + title_height, cv_image.shape[0] - new_y)  # Adjust height to include title
        expanded_boxes.append((new_x, new_y, new_w, new_h))

    # Sort the boxes from top to bottom, then left to right
    expanded_boxes = sorted(expanded_boxes, key=lambda x: (x[1], x[0]))

    # Draw the bounding boxes on the original image for visualization
    for (x, y, w, h) in expanded_boxes:
        cv_image = cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert back to PIL image for display
    output_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    # Display the image with bounding boxes
    plt.imshow(output_image)
    plt.axis('on')
    plt.show()

    # Save individual visuals
    visuals = []
    for i, (x, y, w, h) in enumerate(expanded_boxes):
        visual = image.crop((x, y, x + w, y + h))
        visuals.append(visual)
        visual.save(f"{output_dir}/visual_{i+1}.png")

    # List the saved visuals
    visual_files = [f"{output_dir}/visual_{i+1}.png" for i in range(len(expanded_boxes))]
    return visual_files

def split_dashboard_image_better(image_path, output_dir):
    # Load the image using PIL
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Convert the image to an OpenCV image
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Dilate the image to close gaps in contours
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around each visual
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Filter out small boxes that might not be actual visuals
    min_area = 10000  # Adjust this threshold based on the visuals size
    filtered_boxes = [box for box in bounding_boxes if box[2] * box[3] > min_area]

    # Expand the bounding boxes to include titles, legends, and axes
    expanded_boxes = []
    for (x, y, w, h) in filtered_boxes:
        padding = 20  # Adjust padding as needed
        new_x = max(x - padding, 0)
        new_y = max(y - padding, 0)
        new_w = min(w + 2 * padding, img_width - new_x)
        new_h = min(h + 2 * padding, img_height - new_y)
        expanded_boxes.append((new_x, new_y, new_w, new_h))

    # Sort the boxes from top to bottom, then left to right
    expanded_boxes = sorted(expanded_boxes, key=lambda box: (box[1], box[0]))

    # Draw the bounding boxes on the original image for visualization
    for (x, y, w, h) in expanded_boxes:
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert back to PIL image for display
    output_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    # Display the image with bounding boxes
    plt.imshow(output_image)
    plt.axis('on')
    plt.show()

    # Save individual visuals
    visuals = []
    for i, (x, y, w, h) in enumerate(expanded_boxes):
        visual = image.crop((x, y, x + w, y + h))
        visuals.append(visual)
        visual.save(f"{output_dir}/visual_{i+1}.png")

    # List the saved visuals
    visual_files = [f"{output_dir}/visual_{i+1}.png" for i in range(len(expanded_boxes))]
    return visual_files

def split_dashboard_image_better2(image_path, output_dir):
    # Load the image using PIL
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Convert the image to an OpenCV image
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Dilate the image to close gaps in contours
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around each visual
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Filter out small boxes that might not be actual visuals
    min_area = 10000  # Adjust this threshold based on the visuals size
    filtered_boxes = [box for box in bounding_boxes if box[2] * box[3] > min_area]

    # Expand the bounding boxes to include titles, legends, and axes
    expanded_boxes = []
    for (x, y, w, h) in filtered_boxes:
        padding = 40  # Increase padding to ensure titles are included
        new_x = max(x - padding, 0)
        new_y = max(y - padding, 0)
        new_w = min(w + 2 * padding, img_width - new_x)
        new_h = min(h + 2 * padding, img_height - new_y)
        expanded_boxes.append((new_x, new_y, new_w, new_h))

    # Sort the boxes from top to bottom, then left to right
    expanded_boxes = sorted(expanded_boxes, key=lambda box: (box[1], box[0]))

    # Draw the bounding boxes on the original image for visualization
    for (x, y, w, h) in expanded_boxes:
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert back to PIL image for display
    output_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    # Display the image with bounding boxes
    plt.imshow(output_image)
    plt.axis('on')
    plt.show()

    # Save individual visuals
    visuals = []
    for i, (x, y, w, h) in enumerate(expanded_boxes):
        visual = image.crop((x, y, x + w, y + h))
        visuals.append(visual)
        visual.save(f"{output_dir}/visual_{i+1}.png")

    # List the saved visuals
    visual_files = [f"{output_dir}/visual_{i+1}.png" for i in range(len(expanded_boxes))]
    return visual_files

def split_dashboard_image4(image_path, output_dir):
    # Load the image using PIL
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Convert the image to an OpenCV image
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Dilate the image to close gaps in contours
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around each visual
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Filter out small boxes that might not be actual visuals
    min_area = 10000  # Adjust this threshold based on the visuals size
    filtered_boxes = [box for box in bounding_boxes if box[2] * box[3] > min_area]

    # Dynamically adjust the bounding boxes to include titles, legends, and axes
    adjusted_boxes = []
    for (x, y, w, h) in filtered_boxes:
        top_margin = 10  # Initial small margin to look above the bounding box
        max_height_adjustment = 100  # Max height to look above the box for titles

        # Find the new y-coordinate by scanning upwards for the title
        for dy in range(max_height_adjustment):
            if y - dy <= 0 or np.mean(gray[y - dy:y - dy + 1, x:x + w]) > 240:  # Look for light areas indicating title
                new_y = max(y - dy - top_margin, 0)
                break
        else:
            new_y = max(y - max_height_adjustment - top_margin, 0)

        # Adjust the height to include the new area
        new_h = h + (y - new_y)

        # Include some padding for the sides and bottom
        padding = 10
        new_x = max(x - padding, 0)
        new_w = min(w + 2 * padding, img_width - new_x)
        new_h = min(new_h + padding, img_height - new_y)

        adjusted_boxes.append((new_x, new_y, new_w, new_h))

    # Sort the boxes from top to bottom, then left to right
    adjusted_boxes = sorted(adjusted_boxes, key=lambda box: (box[1], box[0]))

    # Draw the bounding boxes on the original image for visualization
    for (x, y, w, h) in adjusted_boxes:
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert back to PIL image for display
    output_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    # Display the image with bounding boxes
    plt.imshow(output_image)
    plt.axis('on')
    plt.show()

    # Save individual visuals
    visuals = []
    for i, (x, y, w, h) in enumerate(adjusted_boxes):
        visual = image.crop((x, y, x + w, y + h))
        visuals.append(visual)
        visual.save(f"{output_dir}/visual_{i+1}.png")

    # List the saved visuals
    visual_files = [f"{output_dir}/visual_{i+1}.png" for i in range(len(adjusted_boxes))]
    return visual_files

##############################################
def adjust_bounding_boxes(image, boxes):
    adjusted_boxes = []
    for (x, y, w, h) in boxes:
        top_margin = 100  # Maximum margin to look above the bounding box for titles
        new_y = y
        for dy in range(top_margin):
            # Check the average intensity of the row above the bounding box
            if y - dy - 1 >= 0:
                row_intensity = np.mean(image[y - dy - 1, x:x + w])
                if row_intensity > 200:  # Threshold for detecting light-colored text
                    new_y = y - dy
                else:
                    break

        new_h = h + (y - new_y)

        # Include some padding for the sides and bottom
        padding = 10
        new_x = max(x - padding, 0)
        new_w = min(w + 2 * padding, image.shape[1] - new_x)
        new_h = min(new_h + padding, image.shape[0] - new_y)

        adjusted_boxes.append((new_x, new_y, new_w, new_h))

    return adjusted_boxes

def split_dashboard_image(image_path, output_dir):
    # Load the image using PIL
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Convert the image to an OpenCV image
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Dilate the image to close gaps in contours
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around each visual
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Filter out small boxes that might not be actual visuals
    min_area = 10000  # Adjust this threshold based on the visuals size
    filtered_boxes = [box for box in bounding_boxes if box[2] * box[3] > min_area]

    # Adjust the bounding boxes to include titles, legends, and axes
    adjusted_boxes = adjust_bounding_boxes(gray, filtered_boxes)

    # Sort the boxes from top to bottom, then left to right
    adjusted_boxes = sorted(adjusted_boxes, key=lambda box: (box[1], box[0]))

    # Draw the bounding boxes on the original image for visualization
    for (x, y, w, h) in adjusted_boxes:
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert back to PIL image for display
    output_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    # Display the image with bounding boxes
    plt.imshow(output_image)
    plt.axis('on')
    plt.show()

    # Save individual visuals
    visuals = []
    for i, (x, y, w, h) in enumerate(adjusted_boxes):
        visual = image.crop((x, y, x + w, y + h))
        visuals.append(visual)
        visual.save(f"{output_dir}/visual_{i+1}.png")

    # List the saved visuals
    visual_files = [f"{output_dir}/visual_{i+1}.png" for i in range(len(adjusted_boxes))]
    return visual_files

##############################################

# Usage
ROOT_PATH = r"C:\\Users\\NITINS\\OneDrive - Capgemini\\CAPGEMINI\\PROJECT\\GEN AI\\report-usecase\\tableau\\tableau-auth\\pat\\890Portal\\Dashboard_images"
# img_name = 'CMO_Dashboard_Promotion Effect.png'
img_name = 'Recruitment Dashboard_Dashboard 2.png'
folder_name = 'Recruitment Dashboard_Dashboard 2'

if not os.path.exists(os.path.join(ROOT_PATH,folder_name)):
    os.makedirs(os.path.join(ROOT_PATH,folder_name))

image_path = os.path.join(ROOT_PATH,img_name)  # Replace with the path to your image
output_dir = os.path.join(ROOT_PATH,folder_name)  # Replace with the path to your output directory

visual_files = split_dashboard_image_better(image_path, output_dir)
print("Saved visual files:", visual_files)
