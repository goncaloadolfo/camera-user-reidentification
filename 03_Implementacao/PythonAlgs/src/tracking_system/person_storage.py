'''
PersonStorage obj. Used to store entry persons and exit persons.
'''

# libs
import matplotlib.pyplot as plt

__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"


class PersonStorage:

    def __init__(self):
        '''
        Creates and inicializes variables entry and exit lists.
        '''
        self.__entries = []
        self.__exits = []

    def set_entry(self, person):
        '''
        Inserts new entry.

        Args:
        -----
            person (Person) : person to be inserted on entry list
        '''
        self.__entries.append(person)

    def set_exit(self, person):
        '''
        Inserts new exit.

        Args:
        -----
            person (Person) : person to be inserted on exit list
        '''
        self.__exits.append(person)

    def visualize_info(self, event_type="entry", nr_hist=3):
        '''
        Visualize information about detected persons.

        Args:
        -----
            event_type (str) : defines which information to visualize, 'entry' for entries
                and "exit" for exits, default 'entry'
            nr_hist (int) : number of histograms to visualize
        '''
        # get persons list
        persons = self.entries if event_type == "entry" else self.exits
        print("Number of " + event_type + ": ", len(persons))

        # create figures
        plt.figure(event_type + " results")
        plt.subplot(311)
        plt.title("Hue Histograms")
        plt.xlabel("Hue value")
        plt.ylabel("Probability")
        plt.ylim(top=0.3)

        plt.subplot(312)
        plt.title("Saturation Histograms")
        plt.xlabel("Saturation value")
        plt.ylabel("Probability")
        plt.ylim(top=0.3)

        plt.subplot(313)
        plt.title("Area Values")
        plt.xlabel("person")
        plt.ylabel("Area")

        areas = []
        for person_index in range(len(persons)):
            person = persons[person_index]

            # get person info
            person_id = person.id
            area = person.area
            areas.append(area)
            entry_time = person.event_time
            hue_hist = person.hue_hist
            saturation_hist = person.saturation_hist
            hs_2dhist = person.hs_2dhist
            print("id: " + str(person_id) + "; area: " + str(area) + "; " + event_type + " hour: " + str(entry_time))

            if person_index < nr_hist:
                # plot hue histogram
                plt.figure(event_type + " results")
                plt.subplot(311)
                plt.plot(hue_hist)

                # plot saturation histogram
                plt.figure(event_type + " results")
                plt.subplot(312)
                plt.plot(saturation_hist)

                # plot 2d histogram
                plt.figure(event_type + " person " + str(person_index) + " HS histogram")
                plt.title(event_type + " person " + str(person_index) + " HS histogram")
                plt.xlabel("Saturation")
                plt.ylabel("Hue")
                plt.imshow(hs_2dhist, interpolation="nearest")

        # plot area values
        plt.figure(event_type + " results")
        plt.subplot(313)
        plt.plot(areas)

    @property
    def entries(self):
        '''
        Getter for entries list.

        Return:
        -------
            (list) : list of persons who entered
        '''
        return self.__entries

    @property
    def exits(self):
        '''
        Getter for exit list.

        Return:
        -------
            (list) : list of persons who left
        '''
        return self.__exits
