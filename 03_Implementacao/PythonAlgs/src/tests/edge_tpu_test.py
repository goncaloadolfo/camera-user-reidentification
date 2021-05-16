'''
Module to test tracking and matching system.
'''

# built-in
import sys
sys.path.append("..")

# libs
import cv2
import matplotlib.pyplot as plt
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from PIL import ImageDraw
from imutils.video import FPS
from imutils.video import VideoStream
import time

# own modules
from tracking_system.tracking_system import TrackingSystem
from matching_system.brute_force_matcher import BruteForceMatcher
from matching_system.matcher import Matcher
from tracking_system.edge_tpu_method import EdgeTPU
from obfuscation_system.face_blur_edge_tpu import FaceAnonymizer
from database_api.databaseAPI import Database
from imutils.video import VideoStream

__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"

video_capture = cv2.VideoCapture("../../../Dataset/videos/VideoTest3.avi")
video_capture.set(1, 0)

#video_capture = cv2.VideoCapture(0)
#video_capture = VideoStream(usePiCamera=1, resolution=(640, 480)).start()
time.sleep(1)

# create Detection Engine
engine = DetectionEngine("../../models/edge_tpu_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")
#engine = DetectionEngine("../../models/edge_tpu_models/mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite")
labels_path = "../../models/edge_tpu_models/coco_labels.txt"
threshold = 0.5
et_obj = EdgeTPU(engine,labels_path, threshold)
persons_per_frame = []


# create counting system instance
init_point = (0, 200)
end_point = (550, 380)

entry_vector = (0, 1)
counting_system = TrackingSystem(init_point, end_point, entry_vector, et_obj, real_time_track=False, centroid_dist_thr=40)
#                                 tracking_type="nearest_centroid")





while True:
    ret, frame = video_capture.read()
#    frame = video_capture.read()
    
    if frame is None:
        break

    bgr_frame = frame.copy()
    
    # read and process frame
    counting_system.process_frame(None, bgr_frame)
    
    debug_frame = counting_system.debug_frame
    persons = counting_system.last_frame_persons
    
    # draw centroids
    for person in persons:
        centroid = person.centroid
        cv2.putText(frame, "ID " + str(person.id), centroid, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 2)
    persons_per_frame.append(persons)
    
    # draw counting line
    cv2.line(frame, init_point, end_point, (255, 0, 0), thickness=2)
    
    # draw text for number of entries and exitsevent_type + " results"
    entries = counting_system.nr_entries
    exits = counting_system.nr_exists
    cv2.putText(frame, "Entries: " + str(entries), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, "Exits: " + str(exits), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    
    # show original frame, debug frame(person detection) and background image
    cv2.imshow("Frame", debug_frame)
    cv2.imshow("Original frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    
video_capture.release()
#video_capture.stop()
cv2.destroyAllWindows()

# visualize data
person_storage = counting_system.person_storage

print("-- Entries information --")
person_storage.visualize_info(event_type="entry", nr_hist=5)
print("-- Exits information --")
person_storage.visualize_info(event_type="exit", nr_hist=5)
plt.show()

print("-- matching --")
k = 20
thr = sys.float_info.max
n_iter = 500
matcher = BruteForceMatcher(k, thr, n_iter)
matcher.set_person_storage(person_storage)
results = matcher.apply_algorithm()
matcher.write_matches("/home/pi/Desktop/MatchingResults")

print("-- Results --")
Matcher.print_results(results[1], results[2], True)
cv2.destroyAllWindows()

print("Obfuscating...")
# obfuscation system
engine = DetectionEngine("../../models/edge_tpu_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite")
fa = FaceAnonymizer(engine)
video_path = fa.face_anonymizer("output", persons_per_frame, "../../../Dataset/videos/VideoTest3.avi")

## store results on database
print("Storing results on Database...")
servername = "localhost"
database = "Tracking"
username = "root"
password = "projetopass"
db_obj = Database(servername, database, username, password)
db_obj.insert(results, "videoTest3.avi")
print("Done Storing")
