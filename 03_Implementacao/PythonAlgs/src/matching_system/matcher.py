'''
Matching base class.
'''

# built-in
import os

# libs
import matplotlib.pyplot as plt
import numpy as np
import cv2

__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"
FEATURES_NAME = ["area", "hue histogram"]
MATCHES = {20: 7, 51: 32, 37: 25, 29: 1, 48: 34, 30: 9, 53: 41}


class Matcher:

    def __init__(self, thr):
        '''
        Matching constructor.

        Args:
        -----
            thr (float) : matching threshold
        '''

        if type(self) is Matcher:
            raise Exception("Matcher is an abstract class.")

        self.__entries = None
        self.__exits = None
        self.__return_entries = []
        self.__return_exits = []
        self.__corresponded_indexs = []
        self.__thr = thr

    @staticmethod
    def print_results(return_entries, return_exits, person_frames = False):
        '''
        Prints matching results. Return entries/exits lists are
        ordered by matching algorithm.

        Args:
        -----
            return_entries (list): list of entries
            return_exits (list): list of exits
            person_frames (list): True to see persons frames
        '''
        # data structures for matches info
        hue_matches = []
        saturation_matches = []
        hs_matches = []
        cs_matches = []
        edge_matches = []
        area_diffs = []

        # for each correspondence
        for i in range(len(return_exits)):
            entry_person = return_entries[i]
            exit_person = return_exits[i]

            # get correspondence information
            if entry_person is not None:
                entry_id = entry_person.id
                exit_id = exit_person.id
                vector_distance = exit_person.final_match

                # hue match value
                entry_hue_hist = entry_person.hue_hist
                exit_hue_hist = exit_person.hue_hist
                hue_match = Matcher.histogram_match(entry_hue_hist, exit_hue_hist)
                hue_matches.append(hue_match)

                # saturation match value
                entry_saturation_hist = entry_person.saturation_hist
                exit_saturation_hist = exit_person.saturation_hist
                saturation_match = Matcher.histogram_match(entry_saturation_hist, exit_saturation_hist)
                saturation_matches.append(saturation_match)

                # h-s match value
                entry_hs_hist = entry_person.hs_2dhist
                exit_hs_hist = exit_person.hs_2dhist
                hs_match = Matcher.histogram_match(entry_hs_hist, exit_hs_hist)
                hs_matches.append(hs_match)

                # area difference
                entry_area = entry_person.area
                exit_area = exit_person.area
                area_diff = abs(exit_area - entry_area)
                area_diffs.append(area_diff)

                # cs match value
                entry_cs = entry_person.cs_hist
                exit_cs = exit_person.cs_hist
                cs_match = Matcher.histogram_match(entry_cs, exit_cs)
                cs_matches.append(cs_match)

                # edge match value
                entry_edge = entry_person.edge_hist
                exit_edge = exit_person.edge_hist
                edge_match = Matcher.histogram_match(entry_edge, exit_edge)
                edge_matches.append(edge_match)

                # print it
                print("Entry id: " + str(entry_id) + "; " +
                        "Exit id: " + str(exit_id) + "; " +
                        "Vector distance: " + str(vector_distance) + "; " + 
                        "Hue match: " + str(hue_match) + "; " +
                        "Saturation match: " + str(saturation_match) + "; "
                        "H-S match: " + str(hs_match) + "; "
                        "Area diff: " + str(area_diff) + "; "
                        "MPEG7 Color: " + str(cs_match) + ";"
                        "MPEG7 Edges: " + str(edge_match))
                    
                if person_frames:
                    # get persons frames
                    entry_person_frame = entry_person.person_frame
                    exit_person_frame = exit_person.person_frame

                    # resize one of them
                    resized_frame = cv2.resize(exit_person_frame, (entry_person_frame.shape[1], entry_person_frame.shape[0]))

                    # display matching frame
                    matching_frame = np.hstack((entry_person_frame, resized_frame))
                    cv2.imshow("Match " + str(i), matching_frame)

            else:
                print("Exit id: " + str(exit_person.id) + "; no match ;")

                if person_frames:
                    cv2.imshow("Match " + str(i), exit_person.person_frame)

        # plot matches informations
        if person_frames:
            plt.figure("Matching results")

            # Hue matches
            plt.subplot(321)
            plt.title("H Matches")
            plt.ylabel("Match value")
            plt.plot(hue_matches)

            # Saturation matches
            plt.subplot(322)
            plt.title("S Matches")
            plt.ylabel("Match value")
            plt.plot(saturation_matches)

            # HS matches
            plt.subplot(323)
            plt.title("HS Matches")
            plt.ylabel("Match value")
            plt.plot(hs_matches)

            # Area differences
            plt.subplot(324)
            plt.title("Area Difference")
            plt.ylabel("Diff")
            plt.plot(area_diffs)

            # Edge matches
            plt.subplot(325)
            plt.title("MPEG-7 Edge Descriptor")
            plt.xlabel("Match")
            plt.ylabel("Match value")
            plt.plot(edge_matches)

            # Color Structured matches
            plt.subplot(326)
            plt.title("MPEG-7 Color Descriptor")
            plt.xlabel("Match")
            plt.ylabel("Diff")
            plt.plot(cs_matches)

        plt.show()

    @staticmethod
    def histogram_match(hist1, hist2):
        '''
        Calculates histogram match value between two 1D histograms.

        Args:
        -----
            hist1 (ndarray) : histogram 1
            hist2 (ndarray): histogram 2
        '''
        histogram_intersection = np.sum(np.minimum(hist1, hist2))
        match = histogram_intersection * 1.0 / np.sum(hist2)
        return match

    @staticmethod
    def add_missing_values(full_list, incomplete_list):
        '''
        Adds missing values in incomplete list.

        Args:
        -----
            full_list (list) : full list
            incomplete_list (list) : incomplete list
        '''

        inc_list_copy = incomplete_list.copy()

        for aux_p in full_list:
            found = False

            for aux_p2 in inc_list_copy:
                if aux_p2 is not None and aux_p2.id == aux_p.id:
                    found = True
                    break

            if not found:
                incomplete_list.append(aux_p)

    def write_matches(self, write_dir):
        '''
        Write matches frames into a directory

        Args:
        -----
            write_dir (string) : directory path
        '''

        # create dir for results
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        for index in range(len(self.__return_exits)):
            exit_person = self.__return_exits[index]
            entry_person = self.__return_entries[index]

            if entry_person is not None:
                # get persons frames
                entry_person_frame = entry_person.person_frame
                exit_person_frame = exit_person.person_frame

                # resize one of them
                resized_frame = cv2.resize(exit_person_frame,
                                           (entry_person_frame.shape[1], entry_person_frame.shape[0]))

                # display matching frame
                matching_frame = np.hstack((entry_person_frame, resized_frame))

                # write matching frame
                cv2.imwrite(write_dir + "/" + "match " + str(index) + ".jpg", matching_frame)

            # exit person with no match
            else:
                cv2.imwrite(write_dir + "/" + "match " + str(index) + ".jpg", exit_person.person_frame)

    def set_person_storage(self, person_storage):
        '''
        Extracts PersonStorage information.

        Args:
        -----
            person_storage (PersonStorage): obj with person objs
        '''
        self.__entries = person_storage.entries
        self.__exits = person_storage.exits

    def apply_algorithm(self):
        '''
        Applies the matching algorithm between the persons detected on the exit with the persons detected at the entrance.
        In the return list Entry persons / exit persons are lists ordered by matching algoritm. They can have different
        length depending on the detections.

        This is an abstract method!
        '''
        raise NotImplementedError

    def insert_matching(self, exit_person, entry_person, index, match_value):
        '''
        Adds a new match information in their data structures.

        Args:
        -----
            exit_person (Person) : exit person to add
            entry_person (Person) : entry person to add
            index (int) : index to add
            match_value (float) : match histogram value between exit_person and entry_person
        '''
        self.__return_exits.append(exit_person)

        if match_value < self.thr:
            self.__return_entries.append(entry_person)
            self.__corresponded_indexs.append(index)
            exit_person.final_match = match_value

        else:
            self.__return_entries.append(None)
            self.__corresponded_indexs.append(None)
            exit_person.final_match = None

    def reset_results(self):
        '''
        Resets result lists.
        '''
        self.__return_entries = []
        self.__return_exits = []
        self.__corresponded_indexs = []

    def score_test(self):
        '''
        Calculates score for video test.

        Return:
            (float) : score
        '''

        correct_matches = 0
        for ind in range(len(self.return_exits)):
            exit_person = self.return_exits[ind]
            entry_person = self.__return_entries[ind]

            if entry_person is not None:
                if MATCHES[exit_person.id] == entry_person.id:
                    correct_matches += 1

        score = correct_matches * 1.0 / len(self.return_exits)
        return score

    def set_final_results(self, return_entries, return_exits):
        '''
        Updates final result lists.

        Args:
            return_entries (list) : entries list
            return_exits (list) : exits list
        '''
        self.__return_exits = return_exits
        self.__return_entries = return_entries

    @property
    def entries(self):
        '''
        Entries list getter.

        Return:
        -------
            (list) : entries list
        '''
        return self.__entries

    @property
    def exits(self):
        '''
        Exits list getter.

        Return:
        -------
            (list) : exits list
        '''
        return self.__exits

    @property
    def corresponded_indexs(self):
        '''
        Getter for the list with indexes already matched

        Return:
        -------
            (list) : indexes list
        '''
        return self.__corresponded_indexs

    @property
    def thr(self):
        '''
        Matching threshold getter.

        Return:
        -------
            (float) : matching threshold
        '''
        return self.__thr

    @property
    def return_entries(self):
        '''
        Getter for entries matching list.

        Return:
        -------
            (list) : entry persons list
        '''
        return self.__return_entries

    @property
    def return_exits(self):
        '''
        Getter for exits matching list.

        Return:
        -------
            (float) : exit persons list
        '''
        return self.__return_exits
