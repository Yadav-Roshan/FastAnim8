import cv2
import numpy as np
import os

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)

    return equalized

def segment_character(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Apply Gaussian filtering for noise reduction
    blurred = cv2.GaussianBlur(preprocessed_image, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 30, 150)

    # Dilate the edges to close small gaps
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours from the edges
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Remove inner contours and extend contours touching the bottom edge of the image
    mask_height, mask_width = preprocessed_image.shape[:2]

    # Create an empty mask
    mask = np.zeros_like(preprocessed_image)

    # Draw the contours on the mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Fill any small holes or gaps within the contours
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Perform dilation to connect nearby regions and make it one single region
    mask = cv2.dilate(mask, kernel, iterations=2)

    return mask

def add_character_to_background(character_image_path, background_image_path, new_background_image_path):
    # Load the original image and the new background
    original_image = cv2.imread(character_image_path)
    new_background = cv2.imread(background_image_path)

    # Resize the background image to match the size of the original image
    new_background = cv2.resize(new_background, (original_image.shape[1], original_image.shape[0]))

    # Extract the mask for the character
    mask = segment_character(original_image)
    
    mask_inv = cv2.bitwise_not(mask)

    # Extract the character from the original image using the mask
    character = cv2.bitwise_and(original_image, original_image, mask=mask)

    # Extract the background from the new background image using the inverted mask
    background = cv2.bitwise_and(new_background, new_background, mask=mask_inv)

    # Combine the character and background
    result = cv2.add(character, background)

    # Save the result with the character added to the new background
    cv2.imwrite(new_background_image_path, result)
