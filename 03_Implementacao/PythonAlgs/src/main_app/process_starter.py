'''
Script to run an app process.
'''

# built-in
import sys
sys.path.append("..")

import time

# libs
from edgetpu.detection.engine import DetectionEngine
import cv2
from multiprocessing import Event
from imutils.video import VideoStream


# own libs
# from tracking_system.tracking_system import TrackingSystem
from tracking_system.tracking_system_v2 import TrackingSystem
from matching_system.brute_force_matcher import BruteForceMatcher
from database_api.databaseAPI import Database
from main_app import MainApp
from tracking_system.edge_tpu_method import EdgeTPU
#from obfuscation_system.face_blur_edge_tpu import FaceAnonymizer
from obfuscation_system.face_blur_edge_tpu_v2 import FaceAnonymizer

__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"

# create tracking system
# create Detection Engine
#engine = DetectionEngine("mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite")
engine = DetectionEngine("mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")
labels_path = "coco_labels.txt"
threshold = 0.3
pd_obj = EdgeTPU(engine,labels_path, threshold)


init_point = (0, 200)
end_point = (550, 420)
entry_vector = (-220, 550)
centroid_dist = 40
#counting_system = TrackingSystem(init_point, end_point, entry_vector, pd_obj, real_time_track=False,
#                                 tracking_type="iou", real_time_extraction=False, counting_sensor=True, entryPin=21)
counting_system = TrackingSystem(init_point, end_point, entry_vector, pd_obj, real_time_track=False,
                                 tracking_type="iou", real_time_extraction=False)

# create matcher
k = 20
thr = sys.float_info.max
n_iter = 500
matcher = BruteForceMatcher(k, thr, n_iter)

# write definitions
codec = 'XVID'
extension = 'avi'

# local db obj
servername = "localhost"
database = "Tracking"
username = "root"
password = "projetopass"
db_obj = Database(servername, database, username, password)


# create obfuscation system
face_engine = DetectionEngine("../../models/edge_tpu_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite")
person_engine = DetectionEngine("../../models/edge_tpu_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")
fa = FaceAnonymizer(face_engine, person_engine)

# create and run process


#cam =VideoStream(usePiCamera=1, resolution=(640, 480))
#cam= cv2.VideoCapture("videos/CardioID1_Converted.mp4")
cam= cv2.VideoCapture("../../../Dataset/videos/VideoTest3.avi")
#cam= cv2.VideoCapture(0)
app = MainApp(cam,counting_system, matcher, codec, extension, db_obj, fa)
app.run()




