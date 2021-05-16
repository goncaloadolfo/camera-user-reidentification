'''
Tracking system obj.
'''

# libs
import numpy as np
import cv2

# own modules
from matching_system.my_matching import MyMatching
from tracking_system.person_storage import PersonStorage
from tracking_system.dnn_method import DNNMethod

__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"


class TrackingSystem:

    def __init__(self, init_point, end_point, entry_vector, pd_obj, tracking_thr=0.05,
                 real_time_track=False, tracking_type="neareast_iou"):
        '''
        System constructor.

        Args:
        -----
            init_point (tuple) : origin point of counting line
            end_point (tuple) : end point of counting line
            entry_vector (tuple) : entry direction vector
            pd_obj (PersonDetection): obj for person detection
            centroid_dist_thr (int) : threshold for centroid correspondence, default value 20
            real_time_track (bool) : True for real time simple corresponding algoritm, default False
        '''
        self.__init_point = init_point
        self.__end_point = end_point
        self.__pd_obj = pd_obj
        self.__entry_vector = entry_vector
        self.__tracking_thr = tracking_thr
        self.__entries_counter = 0
        self.__exits_counter = 0
        self.__ps_obj = PersonStorage()
        self.__last_frame_persons = []
        self.__frame = None
        self.__bgr_frame = None
        self.__people_counter = 0
        self.__real_time_track = real_time_track
        self.__tracking_type = tracking_type

    @staticmethod
    def intersect(A, B, C, D):
        '''
        Detects if two line segments intersect. It uses cross product method.
        Colinear segments returns False.

        Args:
        -----
            A (tuple): origin of first segment line
            B (tuple) : end of first segment line
            C (tuple) : origin of second segment line
            D (tuple) : end of second segment line

        Return:
        -------
            (bool) : True if AB segment line intersects CD segment line
        '''
        r = (B[0] - A[0], B[1] - A[1])  # AB vector
        s = (D[0] - C[0], D[1] - C[1])  # CD vector
        o = (C[0] - A[0], C[1] - A[1])  # vector between origin of line segments

        cpRS = r[0] * s[1] - r[1] * s[0]  # cross product between AB e CD
        cpOAB = o[0] * r[1] - o[1] * r[0]  # cross product between O and AB
        cpOCD = o[0] * s[1] - o[1] * s[0]  # cross product between O and CD

        if cpRS == 0.0 and cpOAB == 0.0:  # line segments are collinear
            return False

        elif cpRS == 0.0:
            return False  # line segments are parallel

        # A + tR = C + uS
        t = cpOCD / cpRS
        u = cpOAB / cpRS
        return t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0

    @staticmethod
    def match_person(persons, person):
        '''
        Verify which person match to another one.

        Args:
        -----
            persons (list): list of Person who entered
            person (Person): person to match
        '''
        matches = []
        exit_fv = person.get_features_vector()

        # calculate match value for all persons
        for each_entry in persons:
            entry_fv = each_entry.get_features_vector()
            match = np.sum(np.abs(entry_fv - exit_fv))
            matches.append(match)

        # get person with min distance
        matches = np.array(matches)
        min_index = np.argmin(matches)
        match_person = persons[min_index]
        print("Match value: ", matches[min_index])

        cv2.imshow("Match person", match_person.person_frame)
        cv2.waitKey(0)
        cv2.destroyWindow("Match person")

    def process_frame(self, frame, bgr_frame):
        '''
        Dectect people on the frame and check if someone has entered or left
        counting line.

        Args:
        -----
            frame (ndarray) : frame to process
            bgr_frame (ndarray) : original bgr frame
        '''
        self.__bgr_frame = bgr_frame

        if type(self.__pd_obj) is DNNMethod:
            persons = self.__pd_obj.get_persons(bgr_frame)

        else:
            persons = self.__pd_obj.get_persons(frame)

        self.__frame = self.__pd_obj.debug_frame
        # print(str(len(persons)) + " persons detected")
        self.__check_entries_exits(persons)
        self.__last_frame_persons = persons

    def __check_entries_exits(self, persons):
        '''
        Go through each person and:
            - get the correspondence centroid of the previous frame
            - check if segment line of centroids intersects counting line segment line
            - calculate if it was an entry or an exit, increment counters and store person

        Args:
        -----
            (list) : list of people to detect an entry or an exit
        '''
        for person in persons:

            if self.__tracking_type == "nearest_centroid":
                correspondence_person = self.__nearest_person(person)

            else:
                correspondence_person = self.__nearest_iou(person)

            if correspondence_person is None:
                # print("no correspondence")
                person.id = self.__people_counter
                self.__people_counter += 1
                continue

            else:
                person.counted = correspondence_person.counted
                person.id = correspondence_person.id

                if not person.counted:
                    centroid1 = correspondence_person.centroid
                    centroid2 = person.centroid
                    intersect = self.intersect(self.__init_point, self.__end_point, centroid1, centroid2)

                    if intersect:
                        self.__extract_matching_features(person)
                        person.counted = True
                        self.__update_counter_and_storage(centroid1, centroid2, person)

    def __nearest_person(self, person):
        '''
        Calculates distances from person to each person of last frame; get the min
        distance and check if it is lower than threshold.

        Args:
        -----
            person (Person) : person to track with

        Return:
        -------
            (Person) : corresponding person in the previous frame
        '''
        if len(self.__last_frame_persons) == 0:
            return None
        
        else:
            person_centroid = person.centroid
            distances = []
            
            for person in self.__last_frame_persons:
                centroid = person.centroid
                distance = np.sqrt((centroid[0] - person_centroid[0])**2 + (centroid[1] - person_centroid[1])**2)
                distances.append(distance)

            distances = np.array(distances)
            min_distance = np.min(distances)
            index_min = np.argmin(distances)
            # print(minDistance)
            return self.__last_frame_persons[index_min] if np.abs(min_distance) <= self.__tracking_thr else None

    def __update_counter_and_storage(self, old_centroid, new_centroid, person):
        '''
        Checks if it was an entry or an exit, increment the counter and store the person.

        Args:
        -----
            old_centroid (tuple) : centroid of person on previous frame
            new_centroid (tuple) : centroid of person on current frame
            person (Person): person to store
        '''
        movement_vector = (new_centroid[0] - old_centroid[0], new_centroid[1] - old_centroid[1])
        dot_product = movement_vector[0] * self.__entry_vector[0] + movement_vector[1] * self.__entry_vector[1]

        if dot_product > 0:
            self.__entries_counter += 1
            person.set_event_time()
            self.__ps_obj.set_entry(person)
            # print("Entry!")

        else:
            self.__exits_counter += 1
            person.set_event_time()

            entries = self.__ps_obj.entries
            
            if len(entries) != 0 and self.__real_time_track:
                self.match_person(entries, person)
                
            self.__ps_obj.set_exit(person)
            # print("Exit!")

    def __extract_matching_features(self, person):
        '''
        Extracts features used for corresponding process such as
        hue histogram.

        Args:
        -----
            person (Person) : person to extract features
        '''
        person.calc_hists(self.__bgr_frame)
        person.save_person_frame(self.__bgr_frame)
        person.apply_descriptors()

    def __nearest_iou(self, person):
        if len(self.last_frame_persons) == 0:
            return None
        else:

            person_coords = person.coords
            matches = []
            for person in self.last_frame_persons:
                last_coords = person.coords
                iou = self.__iou(last_coords, person_coords)
                matches.append(iou)
            matches = np.array(matches)
            best_iou = np.max(matches)
            index_max = np.argmax(matches)
            return self.__last_frame_persons[index_max] if np.abs(best_iou) >= self.__tracking_thr else None

    def __iou(self, previous_bounding, actual_bounding):

        xs_pre = previous_bounding[1]
        ys_pre = previous_bounding[0]

        xs_act = actual_bounding[1]
        ys_act = actual_bounding[0]

        xA = max(np.min(xs_pre), np.min(xs_act))
        yA = max(np.min(ys_pre), np.min(ys_act))
        xB = min(np.max(xs_pre), np.max(xs_act))
        yB = min(np.max(ys_pre), np.max(ys_act))

        if xB - xA <= 0 or yB - yA <= 0:
            return 0

        intersection = max(0, xB - xA) * max(0, yB - yA)

        box_pre_area = (np.max(xs_pre) - np.min(xs_pre)) * (np.max(ys_pre) - np.min(ys_pre))
        box_act_area = (np.max(xs_act) - np.min(xs_act)) * (np.max(ys_act) - np.min(ys_act))

        iou = intersection / float(box_pre_area + box_act_area - intersection)

        return iou

    @property
    def nr_entries(self):
        '''
        Getter for entries counter.

        Return:
        -------
            (int) : entries counter value
        '''
        return self.__entries_counter

    @property
    def nr_exists(self):
        '''
        Getter for exits counter.

        Return:
        -------
            (int) : exits counter value
        '''
        return self.__exits_counter

    @property
    def debug_frame(self):
        '''
        Getter for the frame with people detection bounding boxes (useless
        for debug).

        Return:
        -------
            (ndarray) : debug frame from people detection obj
        '''
        return self.__frame

    @property
    def last_frame_persons(self):
        '''
        Get people detected in last processed frame.

        Return:
        -------
            (list) : list of persons
        '''
        return self.__last_frame_persons

    @property
    def person_storage(self):
        '''
        Getter for person storage obj.

        Return:
        -------
            (PersonStorage) : person storage obj
        '''
        return self.__ps_obj
