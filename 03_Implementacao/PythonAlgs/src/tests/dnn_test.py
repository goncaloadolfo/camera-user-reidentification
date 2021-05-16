'''
Module to test tracking and matching system.
'''

# built-in
import sys
sys.path.append("..")

# libs
import cv2
import matplotlib.pyplot as plt
import pickle

# own modules
from tracking_system.tracking_system import TrackingSystem
from matching_system.brute_force_matcher import BruteForceMatcher
from matching_system.matcher import Matcher
from tracking_system.dnn_method import DNNMethod
from database_api.databaseAPI import Database
from matching_system.my_matching import MyMatching

__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"

video_capture = cv2.VideoCapture("../../../Dataset/videos/CardioID1_Converted.mp4")
video_capture.set(1, 0)

# create DNN person detection method
net = cv2.dnn.readNetFromCaffe("../../models/MobileNetSSD_deploy.prototxt", "../../models/MobileNetSSD_deploy.caffemodel")
threshold = 0.3
pd_obj = DNNMethod(net, threshold)

# create counting system instance
init_point = (0, 200)
end_point = (500, 420)
entry_vector = (0, 1)
counting_system = TrackingSystem(init_point, end_point, entry_vector, pd_obj, real_time_track=True, tracking_thr=0.05,
                                 tracking_type="iou")

while True:
    ret, frame = video_capture.read()
    
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
cv2.destroyAllWindows()

# visualize data
person_storage = counting_system.person_storage

# with open("person_storage.p", "wb") as output:
#     pickle.dump(person_storage, output, pickle.HIGHEST_PROTOCOL)

# print("-- Entries information --")
# person_storage.visualize_info(event_type="entry", nr_hist=5)
# print("-- Exits information --")
# person_storage.visualize_info(event_type="exit", nr_hist=5)
# plt.show()

# print("-- matching --")
# thr = sys.float_info.max
# k = 20
# n_iter = 10000
# t = 0.0
# matcher = BruteForceMatcher(k, thr, n_iter, t)
# matcher.set_person_storage(person_storage)
# results = matcher.apply_algorithm()

# mymatching = MyMatching(thr)
# mymatching.set_person_storage(person_storage)
# results = mymatching.apply_algorithm()

# matcher.write_matches("C:/MatchingResults")
##print("Score: ", matcher.score_test())

# print("-- Results --")
# Matcher.print_results(results[1], results[2], True)
# cv2.waitKey()
# cv2.destroyAllWindows()

# # obfuscation system
# net = cv2.dnn.readNetFromCaffe("../../models/deploy.prototxt", "../../models/res10_300x300_ssd_iter_140000.caffemodel")
# fa = FaceAnonymizer(net)
# video_path = fa.face_anonymizer("output", "../../../Dataset/videos/VideoTest3.avi")
#
# # store results on database
# servername = "localhost"
# database = "TRACKING"
# username = "root"
# password = "madalena8"
# db_obj = Database(servername, database, username, password)
# db_obj.insert(results, video_path)
