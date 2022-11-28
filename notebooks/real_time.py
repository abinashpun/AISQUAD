from deepface import DeepFace

#DeepFace.stream(db_path='/Users/abinashpun/data_science/AI_Squad/test_ground/test_photos') #opencv
#DeepFace.stream("dataset", detector_backend = 'opencv')
#DeepFace.stream(db_path='/Users/abinashpun/data_science/AI_Squad/test_ground/test_photos', detector_backend = 'ssd')
#DeepFace.stream(db_path='/Users/abinashpun/data_science/AI_Squad/test_ground/test_photos', detector_backend = 'mtcnn')
#DeepFace.stream(db_path='/Users/abinashpun/data_science/AI_Squad/test_ground/test_photos', detector_backend = 'dlib')
DeepFace.stream(db_path='/Users/abinashpun/data_science/AI_Squad/test_ground/test_photos', detector_backend = 'retinaface')
