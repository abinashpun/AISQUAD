from deepface import DeepFace

#DeepFace.stream("dataset", detector_backend = 'opencv')
DeepFace.stream(db_path='test_photos',model_name ='VGG-Face', detector_backend = 'ssd', distance_metric = 'cosine', enable_face_analysis = False, source = 0, time_threshold = 5, frame_threshold = 5 )
#DeepFace.stream(db_path='test_photos', detector_backend = 'mtcnn')
#DeepFace.stream(db_path='test_photos', detector_backend = 'dlib')
#DeepFace.stream(db_path='test_photos', detector_backend = 'retinaface')
