'''
Class to visualize features and compare them between instances
of the same person and diferent persons.
'''

# built-in
import pickle
import sys
sys.path.append("..")

# libs
import matplotlib.pyplot as plt
import numpy as np
import cv2

# own libs

# globals


class FeaturesStudy:

    def __init__(self, pickle_path, target, legend, colors):
        # target info
        self.__target = target
        self.__legend = legend
        self.__colors = colors

        # read pickle
        with open(pickle_path, "rb") as file:
            person_storage = pickle.load(file)

        # get persons
        self.__entries = person_storage.entries
        self.__exits = person_storage.exits
        all_persons = self.__entries + self.__exits

        # features
        self.__areas = []
        self.__hue_hists = []
        self.__sat_hists = []
        self.__hs_hists = []
        self.__edge_hists = []
        self.__cs_hists = []

        for person in all_persons:
            self.__areas.append(person.area)
            self.__hue_hists.append(person.hue_hist)
            self.__sat_hists.append(person.saturation_hist)
            self.__hs_hists.append(person.hs_2dhist)
            self.__edge_hists.append(person.edge_hist)
            self.__cs_hists.append(person.cs_hist)

    @staticmethod
    def plot_nvectors(title, xlabel, ylabel, data, legend=None, colors=None):
        plt.figure()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        for ind in range(len(data)):
            vector = data[ind]

            if colors is not None:
                plt.plot(vector, color=colors[ind])

            else:
                plt.plot(vector)

        if legend is not None:
            plt.legend(legend)

    @staticmethod
    def plot_matrix(title, xlabel, ylabel, matrix):
        plt.figure()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.imshow(matrix)

    def plot_stem(self, correct_distances, wrong_distances, title_suffix):
        plt.figure()
        plt.title("Match distances " + title_suffix)
        plt.xlabel("Entry person")
        plt.ylabel("Distance")
        plt.stem(correct_distances, linefmt='C2-')
        plt.stem(np.arange(len(correct_distances), len(correct_distances) + len(wrong_distances)),
                 wrong_distances, linefmt='C3-')

    def plots_same_person(self):
        features = [self.__areas, self.__hue_hists, self.__sat_hists, self.__hs_hists,
                    self.__edge_hists, self.__cs_hists]
        titles = ["Areas ", "Hue Hists ", "Saturation Hists ", "H-S Histogram ",
                  "Edge Hists ", "Color Structured Hists "]
        xlabels = ["Instance", "Hue", "Saturation", "Saturation", "Edge", "Color"]
        ylabels = ["Area", "Probability", "Probability", "Hue", "Decision Ind", "Decision Ind"]

        for target_id in np.unique(self.__target):
            for feature_ind in range(len(features)):
                target_vectors = np.array(features[feature_ind])[self.__target == target_id]
                target_vectors = target_vectors if feature_ind != 0 else [target_vectors]  # area 2D vector

                # if it is not h-s histograms
                if feature_ind != 3:
                    FeaturesStudy.plot_nvectors(titles[feature_ind] + self.__legend[target_id],
                                                xlabels[feature_ind], ylabels[feature_ind], target_vectors)

                else:
                    for hs_ind in range(len(target_vectors)):
                        matrix = target_vectors[hs_ind]
                        FeaturesStudy.plot_matrix(titles[feature_ind] + self.__legend[target_id] + " " + str(hs_ind),
                                                  xlabels[feature_ind], ylabels[feature_ind], matrix)

            plt.show()

    def comparation_plots(self):
        features = [self.__areas, self.__hue_hists, self.__sat_hists, self.__cs_hists]
        titles = ["Areas", "Hue Hists", "Saturation Hists", "Color Structured Hists"]
        xlabels = ["Instance", "Hue", "Saturation", "Color"]
        ylabels = ["Area", "Probability", "Probability", "Decision ind"]

        for ind in range(len(features)):
            feature = features[ind]
            nvectors = []

            for target_id in np.unique(self.__target):
                target_vectors = np.array(feature)[self.__target == target_id]

                # if it is not area feature
                if ind != 0:
                    mean_vector = np.mean(target_vectors, axis=0)
                    nvectors.append(mean_vector)

                else:
                    nvectors.append(target_vectors)

            FeaturesStudy.plot_nvectors(titles[ind], xlabels[ind], ylabels[ind], nvectors, self.__legend, self.__colors)

        plt.show()

    def plot_distances(self, k, ids_dict):
        # get features vectors
        entry_samples = np.array([person.get_features_vector() for person in self.__entries])
        exit_samples = np.array([person.get_features_vector() for person in self.__exits])

        # calculate distances
        bfm = cv2.BFMatcher()
        matches = bfm.knnMatch(exit_samples, entry_samples, k=k)
        dmatches_list = np.array(matches).ravel()

        # data structures to store distances
        correct_distances = []
        wrong_distances = []

        current_cdistances = []
        current_wdistances = []
        last_person = None

        # for each dmatch
        for dmatch in dmatches_list:
            # get information of the match
            exit_person = self.__exits[dmatch.queryIdx]
            entry_person = self.__entries[dmatch.trainIdx]
            distance = dmatch.distance

            # plot distances if the exit person changes
            if last_person is not None and last_person != exit_person.id:
                self.plot_stem(current_cdistances, current_wdistances, "exit person " + str(exit_person.id))
                current_cdistances = []
                current_wdistances = []
            last_person = exit_person.id

            # same persons
            if ids_dict[exit_person.id] == ids_dict[entry_person.id]:
                current_cdistances.append(distance)
                correct_distances.append(distance)

            # different persons
            else:
                current_wdistances.append(distance)
                wrong_distances.append(distance)

        # calculate threshold
        all_distances = np.array(correct_distances + wrong_distances)

        max_dist = np.max(all_distances)
        min_dist = np.min(all_distances)
        normalized_distances = (all_distances - min_dist) / (max_dist - min_dist)

        threshold = cv2.threshold((normalized_distances * 255.0).astype(np.uint8),
                                  0, 255.0, cv2.THRESH_OTSU)[0]
        threshold = (threshold * (max_dist - min_dist) + min_dist) / 255.0

        # plot all distances
        plt.figure()
        plt.title("All matching distances")
        plt.xlabel("X")
        plt.ylabel("Distance")
        plt.plot(correct_distances, '.', color='g')
        plt.plot(np.arange(len(correct_distances), len(correct_distances) + len(wrong_distances)),
                 wrong_distances, '.', color='r')
        plt.plot(np.arange(len(all_distances)), [threshold] * len(all_distances), 'b')
        plt.show()


if __name__ == "__main__":
    pickle_path = "../../../Dataset/Pickles/person_storage.p"

    target = np.array([0, 3, 2, 1, 2, 0, 3, 1, 2, 3, 0, 2, 0, 3, 1, 0, 2, 3])
    legend = ["Frederico", "Goncalo F", "Goncalo A", "Engenheiro"]
    colors = ["red", "green", "blue", "orange"]

    # FeaturesStudy(pickle_path, target, legend, colors).plots_same_person()
##    FeaturesStudy(pickle_path, target, legend, colors).comparation_plots()

    ids_dict = {7: 'GA', 20: 'GA', 32: 'GA', 51: 'GA', 55: 'GA',
               25: 'GF', 37: 'GF', 44: 'GF',
               1: 'F', 29: 'F', 34: 'F', 48: 'F', 63: 'F',
               9: 'E', 30: 'E', 41: 'E', 53: 'E', 57: 'E'}
    k = 30
    FeaturesStudy(pickle_path, target, legend, colors).plot_distances(k, ids_dict)







