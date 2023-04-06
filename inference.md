# Introduction
This repository contains the code framework for inference part of the face detection. 

# Face Recognition Pipeline
 1. **Face Detection**: At this step, the region of face from photo (captured by webcam or input photo) is detected. The HOG + liner SVM based face detector from `dlib` software is used for the face detction. 

1. **Alignment**: 
The detected face is alined using the face landmark file, `shape_predictor_5_face_landmarks.dat`, provided by [dlib](http://dlib.net/files/).

1. **Normalization**: After alignment, the image pixel is normaized to [0,1] (?).


1. **Representation (Embedding)**: Embedding vector with is created from the normalized detected face. The ArcFace model built from ResNry34 architecture is used to create the embeddings. The pretrained weight for the model is downloaded from [here](https://github.com/serengil/deepface_models/releases) and final model `AISquad_model.h5` is derived. The embedding vector length is 512. 

1. **Classification (Identification)**: Finally, the cosine distance between embedding vector of the detected face and those from employee database is used to identify the detected face. The threshold for the cosine distance is determined with the chefboost decision tree framework.  

6. **Liveliness**: At this stage, we determine the detected face is from the actual person standing infront of the webcam or not. This is work on progress...

