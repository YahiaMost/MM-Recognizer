# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:07:25 2024

@author: Yahia
"""
import cv2
import os
import numpy as np

#%% SIFT
# =============================================================================
#  SIFT Matcher
# =============================================================================
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
symbols_folder = 'High-res'
img2 = cv.imread('High-res-test.png',cv.IMREAD_GRAYSCALE) # trainImage
# img2 = cv.imread('High-res-test2.jpg',cv.IMREAD_GRAYSCALE) # trainImage

symbols = []
for file_name in os.listdir(symbols_folder):
    file_path = os.path.join(symbols_folder, file_name)
    symbol_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if symbol_image is not None:
        symbols.append((file_name, symbol_image))

for (file_name, symbol_image) in symbols:
    img1 = symbol_image

    # Initiate SIFT detector
    sift = cv.SIFT_create()
     
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
     
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
     
    # Apply ratio test
    good = []
    flag = False
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append([m])
            flag=True
            print(file_name, m.distance/n.distance)
    
    # cv.drawMatchesKnn expects list of lists as matches.
    if flag:
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3),plt.show()

#%% FLANN
# =============================================================================
#  FLANN Matcher
# =============================================================================
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
 
# img1 = cv.imread(r'.\\Maker Marks\\image_125.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
symbols_folder = 'High-res'
img2 = cv.imread('High-res-test.png',cv.IMREAD_GRAYSCALE) # trainImage
# img2 = cv.imread('High-res-test2.jpg',cv.IMREAD_GRAYSCALE) # trainImage

symbols = []
for file_name in os.listdir(symbols_folder):
    file_path = os.path.join(symbols_folder, file_name)
    symbol_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if symbol_image is not None:
        symbols.append((file_name, symbol_image))


for (file_name, symbol_image) in symbols:
    img1 = symbol_image
    # Initiate SIFT detector
    sift = cv.SIFT_create()
     
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
     
    flann = cv.FlannBasedMatcher(index_params,search_params)
     
    matches = flann.knnMatch(des1,des2,k=2)
     
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
     
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.5*n.distance:
            matchesMask[i]=[1,0]
    
    draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = cv.DrawMatchesFlags_DEFAULT)
     
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
     
    plt.imshow(img3,),plt.show()