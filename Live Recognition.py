# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 19:48:21 2024

@author: Yahia
"""
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
# Folder path where the symbols are stored
symbols_folder = '.\High-res\\'

# Function to load all the symbols from the folder
def load_symbols(symbols_folder):
    symbols = []
    for file_name in os.listdir(symbols_folder):
        file_path = os.path.join(symbols_folder, file_name)
        symbol_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if symbol_image is not None or symbol_image.any():
            symbols.append((file_name, symbol_image))
    return symbols

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

def detect_symbol_match(camera_frame, symbols):
   sift = cv2.SIFT_create()
   # find the keypoints and descriptors with SIFT
   kp1, des1 = sift.detectAndCompute(camera_frame,None)
    
   for symbol_name, symbol_image in symbols:
       kp2, des2 = sift.detectAndCompute(symbol_image, None)
                 
       # BFMatcher with default params
       bf = cv2.BFMatcher()
       matches = bf.knnMatch(des1, des2, k=2)
        
       # Apply ratio test
       good = []
       for m,n in matches:
           if m.distance < 0.5*n.distance:
               good.append([m])
               print(symbol_name, m.distance/n.distance)
       
# Load all symbols
symbols = load_symbols(symbols_folder)
# Capture and process the camera feed
capture_camera_feed()