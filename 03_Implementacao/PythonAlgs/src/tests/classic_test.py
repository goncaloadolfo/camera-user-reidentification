'''
Module to test tracking and matching system.
'''

# libs
import sys
sys.path.append("..")
import cv2
import numpy as np

# own modules
from tracking_system.tracking_system import TrackingSystem
from tracking_system.classic_method import ClassicMethod
from matching_system.brute_force_matcher import BruteForceMatcher
from matching_system.matcher import Matcher

__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"

video_capture = cv2.VideoCapture("../../../Dataset/videos/VideoTest1.avi")
video_capture.set(1, 0)


# create classic person detection method
def morph_operations_cf(binary_frame):
    dilate_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    return_frame = cv2.dilate(binary_frame.astype(np.uint8), dilate_elem, iterations=3)
    return return_frame


def morph_operations_bs(binary_frame):
    return binary_frame.astype(np.uint8)


background_img = cv2.imread("../../../Dataset/Imgs/BackgroundImg2.tif", cv2.IMREAD_GRAYSCALE)
motion_thr = 30
mp_alpha = 0.97
# pd_obj = ClassicMethod(background_img * 1.0, motion_thr, morph_operations_bs, mp_alpha, True) # background subtraction
pd_obj = ClassicMethod(None, motion_thr, morph_operations_cf, None, False) # consecutive frames subtraction

# create counting system instance
init_point = (310, 300)
end_point = (470, 300)
entry_vector = (0, -1)
counting_system = TrackingSystem(init_point, end_point, entry_vector, pd_obj, real_time_track=False,
                                 tracking_type="nearest_centroid", tracking_thr=40)

while True:
    ret, frame = video_capture.read()
    
    if frame is None:
        break

    bgr_frame = frame.copy()
    
    # read and process frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    counting_system.process_frame(gray_frame, bgr_frame)
    
    debug_frame = counting_system.debug_frame
    persons = counting_system.last_frame_persons
    
    # draw centroids
    for person in persons:
        centroid = person.centroid
        cv2.putText(frame, "ID " + str(person.id), centroid, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 2)
    
    # draw counting line
    cv2.line(frame, init_point, end_point, (255, 0, 0), thickness=2)
    
    # draw text for number of entries and exits
    entries = counting_system.nr_entries
    exits = counting_system.nr_exists
    cv2.putText(frame, "Entries: " + str(entries), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, "Exits: " + str(exits), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    
    # show original frame, debug frame(person detection) and background image
    cv2.imshow("Frame", debug_frame)
    cv2.imshow("Original frame", frame)
    
    bg = pd_obj.auxiliar_frame
    cv2.imshow("Auxiliar frame", bg.astype(np.uint8))

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()

## matching
person_storage = counting_system.person_storage

##print("-- Entries information --")
# person_storage.visualize_info(event_type="entry", nr_hist=5)
##
##print("-- Exits information --")
# person_storage.visualize_info(event_type="exit", nr_hist=5)

# print("-- Matching --")
k = 5
thr = 10
n_iter = 500
matcher = BruteForceMatcher(k, thr, n_iter)
matcher.set_person_storage(person_storage)
results = matcher.apply_algorithm()
#
# print("-- Results --")
Matcher.print_results(results[1], results[2], True)

cv2.waitKey()
cv2.destroyAllWindows()
