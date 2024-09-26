# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:07:25 2024

@author: Yahia
"""

import cv2
import os
import numpy as np

# Load the set of 30 symbols
def load_symbols(symbol_folder):
    symbols = {}
    for filename in os.listdir(symbol_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Assuming image formats
            symbol_name = os.path.splitext(filename)[0]
            symbol_image = cv2.imread(os.path.join(symbol_folder, filename), 0)  # Load in grayscale
            symbols[symbol_name] = symbol_image
    return symbols

# Function to find the best matching symbol using template matching
def find_best_match(frame, symbols):
    best_match = None
    best_score = float('-inf')
    
    frame_height, frame_width = frame.shape[:2]  # Get frame dimensions
    
    for symbol_name, symbol_img in symbols.items():
        if symbol_img is None:
            continue
        symbol_height, symbol_width = symbol_img.shape[:2]  # Get symbol dimensions
        # Resize the symbol if it's larger than the frame
        if symbol_height > frame_height or symbol_width > frame_width:
            scale_factor = min(frame_height / symbol_height, frame_width / symbol_width)
            resized_symbol = cv2.resize(symbol_img, (int(symbol_width * scale_factor), int(symbol_height * scale_factor)))
        else:
            resized_symbol = symbol_img
        
        # Apply template matching with the resized symbol
        res = cv2.matchTemplate(frame, resized_symbol, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > best_score:  # Update best match if current match is better
            best_score = max_val
            best_match = symbol_name

    return best_match, best_score


# Path to the folder containing 30 symbols
symbol_folder = 'Maker Marks'  # Change this to your actual folder path
symbols = load_symbols(symbol_folder)

if not symbols:
    print("No symbols found in the folder.")
    raise

# Start video capture (default camera, use 0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    
    # Convert frame to grayscale for matching
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the best matching symbol
    best_match, best_score = find_best_match(gray_frame, symbols)

    # Display the result on the frame
    if best_match:
        if best_score>=0.7:
            cv2.putText(frame, f"Detected: {best_match} (Score: {best_score:.2f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Real-time Symbol Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

#%%

import cv2
import os

# Folder path where the symbols are stored
symbols_folder = 'Maker Marks'

# Function to load all the symbols from the folder
def load_symbols(symbols_folder):
    symbols = []
    for file_name in os.listdir(symbols_folder):
        file_path = os.path.join(symbols_folder, file_name)
        symbol_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if symbol_image is not None:
            symbols.append((file_name, symbol_image))
    return symbols

# Initialize the camera
def capture_camera_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the captured frame
        cv2.imshow("Camera Feed", frame)

        # Match the symbols with the captured image
        detect_symbol_match(gray_frame, symbols)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Match the camera feed with symbols using ORB feature matching
def detect_symbol_match(camera_frame, symbols):
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors for the camera frame
    kp1, des1 = orb.detectAndCompute(camera_frame, None)

    # Initialize BFMatcher for ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for symbol_name, symbol_image in symbols:
        # Detect keypoints and descriptors for each symbol
        kp2, des2 = orb.detectAndCompute(symbol_image, None)

        if des2 is None:
            continue

        # Match descriptors between camera frame and symbol
        matches = bf.match(des1, des2)

        # Sort matches by distance (lower distance is better)
        matches = sorted(matches, key=lambda x: x.distance)

        # Define a threshold for a good match
        if len(matches) > 10 and matches[0].distance < 50:
            print(f"Match found with symbol: {symbol_name}")
            # Draw matches on the frame (for visualization)
            matched_image = cv2.drawMatches(camera_frame, kp1, symbol_image, kp2, matches[:10], None, flags=2)
            cv2.imshow(f'Match - {symbol_name}', matched_image)

# Load all symbols
symbols = load_symbols(symbols_folder)

# Capture and process the camera feed
capture_camera_feed()

#%%

import cv2
import os
import numpy as np

# Folder path where the symbols are stored
symbols_folder = 'Maker Marks'

# Function to load all the symbols from the folder
def load_symbols(symbols_folder):
    symbols = []
    for file_name in os.listdir(symbols_folder):
        file_path = os.path.join(symbols_folder, file_name)
        symbol_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if symbol_image is not None:
            symbols.append((file_name, symbol_image))
    return symbols

# Initialize the camera
def capture_camera_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the captured frame
        cv2.imshow("Camera Feed", frame)

        # Match the symbols with the captured image
        detect_symbol_match(gray_frame, symbols)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Match the camera feed with symbols using SIFT and FLANN
def detect_symbol_match(camera_frame, symbols):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors for the camera frame
    kp1, des1 = sift.detectAndCompute(camera_frame, None)

    # FLANN parameters for better accuracy and speed
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # Number of times the tree is recursively traversed
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for symbol_name, symbol_image in symbols:
        # Detect keypoints and descriptors for each symbol
        kp2, des2 = sift.detectAndCompute(symbol_image, None)

        if des2 is None or des1 is None:
            continue

        # Find matches using FLANN
        matches = flann.knnMatch(des1, des2, k=2)

        # Store only good matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Define a threshold for a good match
        if len(good_matches) > 10:
            # Extract the matched keypoints from both images
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute the Homography to account for perspective transformation
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # If a valid homography matrix is found, a match is detected
            if M is not None:
                print(f"Match found with symbol: {symbol_name}")

                # Draw matches on the frame (for visualization)
                matches_mask = mask.ravel().tolist()
                h, w = symbol_image.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                # Draw the bounding box on the camera frame
                camera_frame = cv2.polylines(camera_frame, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

                # Show the matched frame
                cv2.imshow(f'Match - {symbol_name}', camera_frame)

# Load all symbols
symbols = load_symbols(symbols_folder)

# Capture and process the camera feed
capture_camera_feed()

#%% From image

import cv2
import os
import numpy as np

# Folder path where the symbols are stored
symbols_folder = 'Maker Marks'

# Path to the independent image
independent_image_path = 'trial.jpg'

# Function to load all the symbols from the folder
def load_symbols(symbols_folder):
    symbols = []
    for file_name in os.listdir(symbols_folder):
        file_path = os.path.join(symbols_folder, file_name)
        symbol_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if symbol_image is not None:
            symbols.append((file_name, symbol_image))
    return symbols

# Load an independent image where the symbols need to be detected
def load_independent_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image from {image_path}")
    return image

# Match the symbols with the independent image using SIFT and FLANN
def detect_symbol_match(independent_image, symbols):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors for the independent image
    kp1, des1 = sift.detectAndCompute(independent_image, None)

    # FLANN parameters for better accuracy and speed
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for symbol_name, symbol_image in symbols:
        # Detect keypoints and descriptors for each symbol
        kp2, des2 = sift.detectAndCompute(symbol_image, None)

        # Skip the symbol if descriptors are empty or None
        if des2 is None or len(kp2) < 2 or des1 is None or len(kp1) < 2:
            print(f"Not enough keypoints found in {symbol_name} or independent image")
            continue

        # Find matches using FLANN
        matches = flann.knnMatch(des1, des2, k=2)

        # Store only good matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Define a threshold for a good match
        if len(good_matches) > 10:
            # Extract the matched keypoints from both images
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute the Homography to account for perspective transformation
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # If a valid homography matrix is found, a match is detected
            if M is not None:
                print(f"Match found with symbol: {symbol_name}")

                # Draw matches on the image (for visualization)
                matches_mask = mask.ravel().tolist()
                h, w = symbol_image.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                # Draw the bounding box on the independent image
                independent_image = cv2.polylines(independent_image, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

                # Show the matched image
                cv2.imshow(f'Match - {symbol_name}', independent_image)

# Load all symbols
symbols = load_symbols(symbols_folder)

# Load the independent image
independent_image = load_independent_image(independent_image_path)

if independent_image is not None:
    # Perform symbol detection
    detect_symbol_match(independent_image, symbols)

    # Display the final image with detected symbols
    cv2.imshow("Detected Symbols", independent_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



