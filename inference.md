# Introduction
This repository contains the code framework for inference part of the face detection. 

# Face Recognition Pipeline
 1. **Face Detection**: 
Detect the region of face from photo (captured by webcam or input photo). The HOG + liner SVM based face detector from dlib is used
1. **Alignment**: 
Align the detected face
1. **Normalization**:
 Normalize the image pixel (face)
1. **Representation (Embedding)**:

## Classification
 (i.e. identifying person correspondig to face)