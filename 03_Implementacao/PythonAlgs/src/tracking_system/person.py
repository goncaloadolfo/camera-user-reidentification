'''
Person object.
'''

# built-in
import datetime

# libs
import numpy as np
from sklearn.preprocessing import StandardScaler

# own modules
from matching_system.my_matching import MyMatching
from tracking_system.feature_extraction import FeatureExtraction


__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"
NUM_COLORS = 64
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'


class Person:

    def __init__(self, centroid, coords):
        '''
        Person constructor.

        Args:
        -----
            centroid (tuple) : person centroid (x, y)
            coords (ndarray) : coordinates of the person
        '''
        # info
        self.__centroid = centroid
        self.__coords = coords
        self.__id_person = None
        self.__event_time = None
        self.__counted = False
        self.__person_frame = None

        # features
        self.__hue_hist = None
        self.__saturation_hist = None
        self.__hs_2dhist = None
        self.__edge_hist = None
        self.__cs_hist = None

        # matching alg attbs
        self.__final_match = None
        self.__match_values = []
        self.__possible_correspondences = []

    def calc_hists(self, frame):
        '''
        Calculates hue and saturation histogram.

        Args:
        -----
            frame (ndarray) : bgr frame where person was detected
        '''
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        mask[self.__coords] = 255
        self.__hue_hist = FeatureExtraction.my_calc_hist(frame, channel=[0], mask=mask, bins=[180], ranges=[0, 180])
        self.__saturation_hist = FeatureExtraction.my_calc_hist(frame, channel=[1], mask=mask, bins=[256], ranges=[0, 256])
        self.__hs_2dhist = FeatureExtraction.my_calc_hist(frame, channel=[0, 1], mask=mask, bins=[10, 10], ranges=[0, 180, 0, 256])

    def save_person_frame(self, frame):
        '''
        Saves person frame.

        Args:
        -----
            frame (ndarray) : bgr frame where person was detected
        '''
        xs = self.__coords[1]
        ys = self.__coords[0]
        
        x_max = np.max(xs)
        x_min = np.min(xs)
        y_max = np.max(ys)
        y_min = np.min(ys)
        
        self.__person_frame = frame[y_min:y_max, x_min:x_max, :]

    def set_event_time(self):
        '''
        Record event time.
        '''
        self.__event_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
    
    def calc_possible_matches(self, persons):
        '''
        Calculates match histogram value for each possible
        person in the list received. Returns the max match value
        and the index of the person in the list.

        Args:
        -----
            persons (list) : list of persons

        Return:
        -------
            (tuple) : match value and its index
        '''
        for index_person in range(len(persons)):
            person = persons[index_person]
            entry_time = person.event_time
            exit_time = self.__event_time

            # if it is a possible match
            if exit_time > entry_time:
                match_value = MyMatching.histogram_match(self.__hue_hist, person.hue_hist)
                self.__match_values.append(match_value)
                self.__possible_correspondences.append(index_person)

        self.__match_values = np.array(self.__match_values)
        self.__possible_correspondences = np.array(self.__possible_correspondences)

        # if the number of possible matches is non-zero
        if len(self.__match_values) != 0:
            max_index = np.argmax(self.__match_values)
            
            return self.__match_values[max_index], self.__possible_correspondences[max_index]

    def get_alternative_match(self, corresponded_indexs):
        '''
        Returns the next non-corresponded index and corresponding match value.

        Args:
        -----
            corresponded_indexs (list) : list of indexs alredy corresponded

        Return:
        -------
            (tuple) : match value and its index
        '''
        # sort lists by match values
        ord_matches = self.__match_values[np.argsort(-self.__match_values)]
        ord_indexs = self.__possible_correspondences[np.argsort(-self.__match_values)]

        # get the next highest possible match
        for aux in range(len(ord_matches)):
            if not (ord_indexs[aux] in corresponded_indexs):
                return ord_matches[aux], ord_indexs[aux]

    def get_match_value(self, index):
        '''
        Get the match value of a specific index.

        Args:
        -----
            index (int) : person index

        Return:
        -------
            (float) : match value
        '''
        return self.__match_values[self.__possible_correspondences == index]

    def apply_descriptors(self):
        '''
        Apply edge and color descriptors to person frame.
        '''
        self.__edge_hist = FeatureExtraction.ehd(self.__person_frame)
        self.__cs_hist = FeatureExtraction.csd(self.__person_frame, NUM_COLORS)

    @property
    def final_match(self):
        '''
        Getter for final match value.

        Return:
        -------
            (float) : final match value
        '''
        return self.__final_match

    @final_match.setter
    def final_match(self, match_value):
        '''
        Setter for final match value.

        Args:
        -----
            match_value (float) : match value to set
        '''
        self.__final_match = match_value

    @property
    def max_match(self):
        '''
        Get max possible match value.

        Return:
        -------
            (float) : match value
        '''
        return np.max(self.__match_values)

    @property
    def counted(self):
        '''
        Getter for counted att: says whether it has already been counted or not.

        Return:
        -------
            (bool) : counted value
        '''
        return self.__counted

    @counted.setter
    def counted(self, counted_value):
        '''
        Setter for counted value.

        Args:
        -----
            (bool) : counted value
        '''
        self.__counted = counted_value

    def get_features_vector(self):
        '''
        Get features vector of person.

        Return:
        -------
            (ndarray) : features vector of dtype float32 and shape (1, n features)
        '''
        area = np.array([self.area])[:, np.newaxis]
        features_vector = np.concatenate((area, self.hue_hist, self.saturation_hist, self.cs_hist[:, np.newaxis]))
        return StandardScaler().fit_transform(features_vector.astype("float32"))

    @property
    def coords(self):
        return self.__coords

    @property
    def centroid(self):
        '''
        Getter for person centroid.

        Return:
        -------
            (tuple) : person centroid
        '''
        return self.__centroid

    @property
    def hue_hist(self):
        '''
        Getter for hue histogram.

        Return:
            (ndarray) : hue histogram
        '''
        return self.__hue_hist

    @property
    def area(self):
        '''
        Getter for area value.

        Return:
            (int): person area
        '''
        return len(self.__coords[0])

    @property
    def id(self):
        '''
        Getter for id.

        Return:
        -------
            (int) : person id
        '''
        return self.__id_person

    @id.setter
    def id(self, id_person):
        '''
        Setter for id.

        Args:
        -----
            id_person (int) : id to set
        '''
        self.__id_person = id_person

    @property
    def event_time(self):
        '''
        Getter for event time.

        Return:
        -------
            (str) : event time
        '''
        return self.__event_time

    @property
    def person_frame(self):
        '''
        Getter for person frame.

        Return:
        -------
            (ndarray) : person frame
        '''
        return self.__person_frame.astype(np.uint8)

    @property
    def saturation_hist(self):
        '''
        Getter for saturation histogram.

        Return:
        -------
            (ndarray) : saturation histogram
        '''
        return self.__saturation_hist

    @property
    def hs_2dhist(self):
        '''
        Getter for 2d hue and saturation histogram.

        Return:
        -------
            (ndarray) : h-s histogram
        '''
        return self.__hs_2dhist

    @property
    def edge_hist(self):
        '''
        Getter edge histogram.

        Return:
        -------
            (ndarray) : edge histogram
        '''
        return self.__edge_hist

    @property
    def cs_hist(self):
        '''
        Getter for color structured histogram.

        Return:
        -------
            (ndarray) : color structured histogram
        '''
        return self.__cs_hist
