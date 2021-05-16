'''
Brute force matching algorithm.
'''

# built-in
from random import shuffle
from datetime import datetime

# libs
import numpy as np
import cv2

# own libs
from matching_system.matcher import Matcher, FEATURES_NAME
from tracking_system.person import DATETIME_FORMAT

__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"
NO_MATCH_COST = 100000


class BruteForceMatcher(Matcher):

    def __init__(self, k, thr, n_iter, t=0.0):
        '''
        Calls super constructor to make all initializations.

        Args:
        -----
            k (int) : number of matches to find
            thr (float) : threshold to be considered a match
            iter (int) : number of algorithm iterations
            t (float) : match temporal weight, default 0.0
        '''
        super(BruteForceMatcher, self).__init__(thr)
        self.__k = k
        self.__iter = n_iter
        self.__t = t

    @staticmethod
    def random_indexes(length):
        '''
        Generates a numpy array of non-repeated integers with 'length' dimension.
        Values between 0 - ('length' - 1)

        Args:
        -----
            length (int) : length of the output array / max value

        Return:
        -------
            (ndarray) : int array
        '''
        range_list = [i for i in range(length)]
        shuffle(range_list)
        return range_list

    def apply_algorithm(self):
        '''
        Applies the bruteforce algorithm for 'iter' iterations. Returns the results in the iteration with lowest cost.

        Return:
        -------
            (list) : [features name, entry persons, exit persons] format
        '''
        costs = []
        results = []

        for it_number in range(self.__iter):
            # reset result lists
            self.reset_results()

            print("#iteration ", it_number + 1)
            # run algorithm
            self.__run_iter()
            # Matcher.print_results(self.return_entries, self.return_exits)

            # get results
            entries = self.return_entries.copy()
            exits = self.return_exits.copy()

            # calculate cost
            cost = self.__calc_cost()
            print("Distances sum: ", cost)

            # append cost and results
            costs.append(cost)
            results.append([entries, exits])

        costs = np.array(costs)
        best_iter = np.argmin(costs)
        print("Best iteration: ", costs[best_iter])
        return_entries = results[best_iter][0]
        return_exits = results[best_iter][1]

        # update result attbs
        super(BruteForceMatcher, self).set_final_results(return_entries, return_exits)

        return [FEATURES_NAME, return_entries, return_exits]

    def write_matches(self, write_dir):
        '''
        Calls super write_matches method.
        '''
        super(BruteForceMatcher, self).write_matches(write_dir)

    def score_test(self):
        '''
        Calculates score for video test.

        Return:
            (float) : score
        '''
        return super(BruteForceMatcher, self).score_test()

    def __best_dmatch(self, dmatches):
        '''
        Receives a list of DMatches and returns the best possible dmatch according to event times and already
        corresponded entry persons.

        Args:
        -----
            dmatches (list) : list of dmatches

        Return:
        -------
            (bool) : True if it founds valid match
            (DMatch or int): if it founds a valid match returns 
                dmatch, otherwise index of query sample
        '''
        for dmatch in dmatches:
            exit_idx = dmatch.queryIdx
            entry_idx = dmatch.trainIdx

            if self.__is_valid(exit_idx, entry_idx):
                return True, dmatch

        return False, dmatches[0].queryIdx

    def __is_valid(self, exit_idx, entry_idx):
        '''
        Checks if it is a valid match.

        Args:
        -----
            exit_idx (int) : index of person on exit persons list
            entry_idx (int) : index of person on entry persons list

        Return:
        -------
            (bool) : True if it is a valid match else False
        '''
        exit_person = self.exits[exit_idx]
        entry_person = self.entries[entry_idx]

        entry_time = entry_person.event_time
        exit_time = exit_person.event_time

        return exit_time > entry_time and entry_idx not in self.corresponded_indexs

    def __calc_cost(self):
        '''
        Gets algorithm results and calculates iteration cost.

        Return:
        -------
            (float) : cost
        '''
        distances_sum = 0.0

        for ind in range(len(self.return_exits)):
            exit_person = self.return_exits[ind]
            distance = exit_person.final_match

            # match
            if distance != -1:
                entry_person = self.return_entries[ind]
                exit_person_time = datetime.strptime(exit_person.event_time, DATETIME_FORMAT)
                entry_person_time = datetime.strptime(entry_person.event_time, DATETIME_FORMAT)
                temporal_diference = (exit_person_time - entry_person_time).seconds / 1000.0
                d = (1 - self.__t) * distance + self.__t * temporal_diference
                distances_sum += d

            # no match
            else:
                distances_sum += NO_MATCH_COST

        return distances_sum

    def __run_iter(self):
        '''
        Applies the bruteforce algorithm. Find the best k matches and the match will be the best possible match
        according to distance metrics. The exit persons list is randomly iterated. In the return list
        entry persons / exit persons are lists ordered by matching algoritm. They can have different
        length depending on the detections.
        '''
        # create bf matcher instance
        bfm = cv2.BFMatcher()

        # get features vectors
        entry_samples = np.array([person.get_features_vector() for person in self.entries])
        exit_samples = np.array([person.get_features_vector() for person in self.exits])

        # calculate k matches
        matches = bfm.knnMatch(exit_samples, entry_samples, k=self.__k)

        # get random indexes to iterate on
        order = BruteForceMatcher.random_indexes(exit_samples.shape[0])

        for idx in order:  # random indexs
            dmatches_list = matches[idx]
            match = self.__best_dmatch(dmatches_list)
            found = match[0]

            # if there is a possible match and its distance is lower than a thr
            if found:
                exit_person = self.exits[match[1].queryIdx]
                entry_person = self.entries[match[1].trainIdx]
                distance = match[1].distance
                self.insert_matching(exit_person, entry_person, match[1].trainIdx, distance)

            else:
                query_idx = match[1]
                exit_person = self.exits[query_idx]
                self.insert_matching(exit_person, None, None, -1)

        # insert entry persons who dont have a match
        Matcher.add_missing_values(self.entries, self.return_entries)
