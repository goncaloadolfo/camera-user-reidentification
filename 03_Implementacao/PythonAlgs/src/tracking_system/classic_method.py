'''
Classic motion detection methodology.
'''

# libs
import cv2
import numpy as np

# own modules
from tracking_system.person import Person
from tracking_system.person_detection import PersonDetection

__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"


class ClassicMethod(PersonDetection):
    
    def __init__(self, auxiliar_frame, motion_thr, morph_operations, bg_alpha, method):
        '''
        Class constructor.

        Args:
        -----
            auxiliar_frame (ndarray): background image in case of
                background subtraction method, None otherwise
            motion_thr (int): motion threshold
            morph_operations (Method) : method that applies morphlogy operations
            bg_alpha (float): alpha value for background update, ignored if
                subtraction of consecutive frames is used
            method (bool): True for background subtraction method, otherwise subtraction
                of consecutive frames
        ''' 
        self.__auxiliar_frame = auxiliar_frame
        self.__motion_thr = motion_thr
        self.__morph_operations = morph_operations
        self.__frame = None
        self.__bg_alpha = bg_alpha
        self.__method = method

    def get_persons(self, frame):
        '''
        Detects people in a frame.

        Args:
        -----
            frame (ndarray): frame that is intended to detect people

        Return:
        -------
            (list) : list of people detected
        '''

        motion_pixels = self.__get_motion_pixels(frame)
        motion_pixels_mo = self.__morph_operations(motion_pixels)
        self.__frame = motion_pixels_mo
        label_frame = self.get_connected_regions(motion_pixels_mo)
        
        labels = np.unique(label_frame)
        labels = labels[labels != 0]
        persons = []
        for label in labels:
            area = np.sum(label_frame == label)
            
            if area > 400:
                coords = np.where(label_frame == label)
                xs = coords[1]
                ys = coords[0]
                
                x_max = np.max(xs)
                x_min = np.min(xs)
                y_max = np.max(ys)
                y_min = np.min(ys)
                
                mean_x = np.mean(xs)
                mean_y = np.mean(ys)
                centroid = (int(mean_x), int(mean_y))
                persons.append(Person(centroid, coords))
                
                if self.__frame is not None:
                    cv2.rectangle(self.__frame, 
                                  (x_min, y_min), 
                                  (x_max, y_max,), 
                                  (255, 255, 255), 2)
        
        self.__update_auxiliar_frame(motion_pixels_mo, frame)
        
        return persons

    @staticmethod
    def get_connected_regions(bin_frame):
        '''
        Extract label frame.

        Args:
        -----
            bin_frame (ndarray) : binary frame(0/255)

        Return:
        -------
            (ndarray) : label frame
        '''
        return cv2.connectedComponents(bin_frame)[1]

    @staticmethod
    def calc_background_img(video_path, frames_thr, write_path):
        '''
        Calculate initial background image of a video.
        Reads the first framesThr frames and makes a temporal median.

        Args:
        -----
            video_path (str) : path for video
            frames_thr (int) : number of frames to be considered
            write_path (str) : path to the background image
        '''
        cap_obj = cv2.VideoCapture(video_path)

        # video information
        total_frames = int(cap_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total frames: ", total_frames)
        nr_rows = int(cap_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nr_columns = int(cap_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Resolução das frames: " + str(nr_rows) + "x" + str(nr_columns))

        frames = []
        frames_thr = frames_thr if frames_thr < total_frames else total_frames
        for _ in range(frames_thr):
            ret, frame = cap_obj.read()

            if frame is None:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame)

        frames = np.array(frames)
        background_img = (np.median(frames, axis=0)).astype(np.uint8)
        cv2.imshow("Background image ", background_img)
        cv2.imwrite(write_path, background_img)

        cap_obj.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @property
    def debug_frame(self):
        '''
        Getter for debug frame.

        Return:
        -------
            (ndarray) : frame with people bounding boxes for debug
        '''
        return self.__frame

    @property
    def auxiliar_frame(self):
        '''
        Getter for current background image.

        Return:
        -------
            (ndarray) : current background frame
        '''
        return self.__auxiliar_frame

    def __get_motion_pixels(self, gray_frame):
        '''
        Extract motion pixels.

        Args:
        -----
            gray_frame (ndarray) : frame that is intended to extract motion pixels

        Return:
        -------
            (ndarray) : binary frame(0/255) of motion pixels
        '''
        if self.__auxiliar_frame is None:
            self.__auxiliar_frame=gray_frame
            
        dif_frame = np.abs(gray_frame * 1.0 - self.__auxiliar_frame * 1.0)
        return cv2.threshold(dif_frame, self.__motion_thr, 255.0, cv2.THRESH_BINARY)[1]

    def __update_auxiliar_frame(self, motion_pixel_frame, gray_frame):
        '''
        Updates background image.

        Args:
        -----
            motion_pixel_frame (ndarray) : binary frame(0/255) with motion pixels
            gray_frame (ndarray) : corresponding gray frame of motion pixels frame
        '''
        if self.__method:
            bg_aux = self.__auxiliar_frame * 1.0 * self.__bg_alpha + gray_frame * 1.0 * (1.0 - self.__bg_alpha)
            not_mp_coords = np.where(motion_pixel_frame == 0)
            self.__auxiliar_frame[not_mp_coords] = bg_aux[not_mp_coords]
            
        else:
            self.__auxiliar_frame=gray_frame

