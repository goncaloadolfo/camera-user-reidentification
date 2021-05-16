'''
EdgeTPU detection methodology.
'''

# libs
import cv2
import numpy as np

# own modules
from tracking_system.person import Person
from tracking_system.person_detection import PersonDetection
from edgetpu.detection.engine import DetectionEngine

from edgetpu.detection.engine import DetectionEngine
from PIL import Image
import PIL

__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"


class EdgeTPU(PersonDetection):

    def __init__(self, engine, label_path, threshold=0.3):
        self.__engine = engine
        self.__labels = self.__readLabelFile(label_path)
        self.__threshold = threshold
        self.__frame = None

    def __detect_persons(self, bgr_frame):
       
        image = Image.fromarray(bgr_frame)
    

        # Run Inference
        # threshold - float defining the minimun confidence threshold
        # top_k - int defining maximum number of objects to detect
        # relative_coord - bool defining if returns float coords or int
        results = self.__engine.DetectWithImage(image, threshold=self.__threshold, keep_aspect_ratio=True,  #PRESO AQUI
                                                relative_coord=False, top_k=10, resample=PIL.Image.BICUBIC)
        
        return results

    def get_persons(self, og_frame):
        '''
        Abstract method!
        Detects persons in received frame.

        Args:
        -----
            frame (ndarray) : frame intended to detect people

        Return:
        -------
            (list) : list of persons detected
        '''

        persons = []
        self.__frame = og_frame.copy()
        
        ans = self.__detect_persons(self.__frame)
       
        if ans:
            for obj in ans:
                if obj.label_id == 0:
                    score = obj.score * 100
                    bounding_box = obj.bounding_box.flatten().tolist()
                    a_x = int(bounding_box[0])
                    a_y = int(bounding_box[1])
                    b_x = int(bounding_box[2])
                    b_y = int(bounding_box[3])

                    cx = int((a_x + b_x) / 2)
                    cy = int((a_y + b_y) / 2)

                    cv2.circle(self.__frame, (cx, cy), 5, (0, 255, 0), -1)

                    aux = np.zeros((self.__frame.shape[0], self.__frame.shape[1]), dtype=np.uint8)
                    aux[a_y: b_y, a_x: b_x] = 255
                    coords = np.where(aux == 255)

                    person = Person((cx, cy), coords)
                    persons.append(person)

                    cv2.rectangle(self.__frame, (a_x, a_y), (b_x, b_y), (255, 0, 0), 2)
                    cv2.putText(self.__frame, str(score), (a_x, a_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        return persons

    def __readLabelFile(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        ret = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            ret[int(pair[0])] = pair[1].strip()
        return ret

    @property
    def debug_frame(self):
        '''
        Abstract method!
        Obtains a debug frame with persons bounding boxes.

        Return:
        -------
            (ndarray) : frame
        '''
        return self.__frame
