'''
DNN detection methodology.
'''

# libs
import cv2
import numpy as np

# own modules
from tracking_system.person import Person
from tracking_system.person_detection import PersonDetection

__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"


class DNNMethod(PersonDetection):

    def __init__(self, net, threshold):
        self.__net = net
        self.__frame = None
        self.__threshold = threshold

    def get_persons(self, og_frame):
        '''
        Detects faces on the frame
        :param 2D uint8 numpy array frame
        :return 4D uint8 numpy array: exits counter value
        '''
        self.__frame = og_frame.copy()

        # executa subtracao de media, e escala imagem
        frame_resize = cv2.resize(self.__frame, (300, 300))
        blob = cv2.dnn.blobFromImage(frame_resize, 0.007843,
                                     # blob - coleccao de imagens com as mesmas dimensoes
                                     (300, 300), (127.5, 127.5, 127.5),
                                     False)  # argumentos - frame, scale, cnn spatialsize, media R-G-B
        self.__net.setInput(blob)
        detected_objects = self.__net.forward()  # forward pass - proceso de calculo input para output layers
        persons = []

        frame_shape = frame_resize.shape[:2]
        h = frame_shape[0]
        w = frame_shape[1]

        for i in range(detected_objects.shape[2]):

            # probabilidade de ser face
            probability = detected_objects[0, 0, i, 2]

            if probability > self.__threshold:
                id_class_person = int(detected_objects[0, 0, i, 1])  # Class label

                if id_class_person == 15:  # label 15 refers to class person
                    bounding_box = detected_objects[0, 0, i, 3:7] * np.array([w, h, w, h])

                    (a_x, a_y, b_x, b_y) = bounding_box.astype("int")

                    # Scaling to original frame size
                    h_factor = self.__frame.shape[0] / 300
                    w_factor = self.__frame.shape[1] / 300

                    # Scaling bounding box to current frame
                    a_x = int(w_factor * a_x)
                    a_y = int(h_factor * a_y)
                    b_x = int(w_factor * b_x)
                    b_y = int(h_factor * b_y)

                    # limit coords
                    a_x = a_x if a_x > 0 else 0
                    a_y = a_y if a_y > 0 else 0
                    b_x = b_x if b_x < self.__frame.shape[1] else self.__frame.shape[1]
                    b_y = b_y if b_y < self.__frame.shape[0] else self.__frame.shape[0]

                    cv2.rectangle(self.__frame, (a_x, a_y), (b_x, b_y), (255, 0, 0), 2)
                    cv2.putText(self.__frame, str(probability), (a_x, a_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

                    cx = int((a_x + b_x) / 2)
                    cy = int((a_y + b_y) / 2)

                    cv2.circle(self.__frame, (cx, cy), 5, (0, 255, 0), -1)

                    aux = np.zeros((self.__frame.shape[0], self.__frame.shape[1]), dtype=np.uint8)
                    aux[a_y: b_y, a_x: b_x] = 255
                    coords = np.where(aux == 255)
                    
                    person = Person((cx, cy), coords)
                    persons.append(person)

        return persons

    @property
    def debug_frame(self):
        '''
        Getter for debug frame.

        Return:
        -------
            (ndarray) : frame with people bounding boxes for debug
        '''
        return self.__frame
