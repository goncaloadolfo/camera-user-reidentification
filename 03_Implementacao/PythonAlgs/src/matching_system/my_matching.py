'''
Matching algorithm based on histogram intersections.
'''

# own libs
from matching_system.matcher import Matcher

__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"
FEATURES_NAME = ["area", "hue histogram"]


class MyMatching(Matcher):

    def __init__(self, thr):
        '''
        Matching constructor.

        Args:
        -----
            thr (float) : matching threshold
        '''
        super(MyMatching, self).__init__(thr)

    def apply_algorithm(self):
        '''
        Applies the matching algorithm between the persons detected on the exit with the persons detected at the entrance.
        In the return list Entry persons / exit persons are lists ordered by matching algoritm. They can have different
        length depending on the detections.

        Return:
        -------
            (list) : a list with structure [features name, entry persons, exit persons]
        '''
        for exit_person in self.exits:
            # get the index corresponding to the person with highest match value
            result = exit_person.calc_possible_matches(self.entries)

            if result is not None:
                match_value = result[0]
                max_correspondence = result[1]

                # if it doesnt have already a match with that person
                if not (max_correspondence in self.corresponded_indexs):
                    self.insert_matching(exit_person, self.entries[max_correspondence],
                                         max_correspondence, match_value)

                # if it does
                else:
                    self.__matching_collision(exit_person, match_value, max_correspondence)

        Matcher.add_missing_values(self.entries, self.return_entries)

        return [FEATURES_NAME, self.return_entries, self.return_exits]

    def score_test(self):
        '''
        Calculates score for video test.

        Return:
            (float) : score
        '''
        return super(MyMatching, self).score_test()

    def __matching_collision(self, exit_person, match_value, index):
        '''
        Method called each time a matching collision is detected. Matches are formatted based on the alternatives
        and its costs.

        Args:
        -----
            exit_person (Person) : person who caused collision
            match_value (float) : match histogram value between exit_person and entries[max_correspondence]
            index (int) : index of the entry person who obtained the highest match value
        '''
        # get alternatives
        collision_index = self.corresponded_indexs.index(index)
        collision_person = self.exits[collision_index]

        alt_result1 = exit_person.get_alternative_match(self.corresponded_indexs)
        alt_result2 = collision_person.get_alternative_match(self.corresponded_indexs)

        # if both have alternatives
        if alt_result1 is not None and alt_result2 is not None:
            match_alt1, index_alt1 = alt_result1
            match_alt2, index_alt2 = alt_result2

            g1 = match_alt1 + collision_person.get_match_value(index)
            g2 = match_alt2 + match_value

            # check which alternative is the best
            if g1 > g2:
                self.insert_matching(exit_person, self.entries[index_alt1], index_alt1, match_alt1)

            else:
                self.insert_matching(exit_person, self.entries[index], index, match_value)

                self.return_entries[collision_index] = self.entries[index_alt2]
                self.corresponded_indexs[collision_index] = index_alt2
                collision_person.final_match = match_alt2

        # one of the persons have alternatives
        elif alt_result1 is not None:
            match_alt1, index_alt1 = alt_result1
            self.insert_matching(exit_person, self.entries[index_alt1], index_alt1, match_alt1)

        elif alt_result2 is not None:
            match_alt2, index_alt2 = alt_result2
            self.insert_matching(exit_person, self.entries[index], index, match_value)

            self.return_entries[collision_index] = self.entries[index_alt2]
            self.corresponded_indexs[collision_index] = index_alt2
            collision_person.final_match = match_alt2

        # None of the persons have alternatives
        else:
            if match_value > collision_person.get_match_value(index):
                self.return_exits[collision_index] = exit_person
                exit_person.final_match = match_value

                self.insert_matching(collision_person, None, None, -1)
                collision_person.final_match = None

            else:
                self.insert_matching(exit_person, None, None, -1)


