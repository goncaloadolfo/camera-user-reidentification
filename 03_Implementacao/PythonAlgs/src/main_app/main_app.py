'''
Main application class.
'''

# built-in
import time
import os
import socket


# libs
import cv2
from imutils.video import FPS
from imutils.video import VideoStream
import urllib3
from database_api.amazonAPI import Amazon_API
import numpy as np
import logging

__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"

# dirs
VIDEOS_TEMP_DIR = "/home/pi/Desktop/TemporaryVideos"
MATCHING_DIR = "/home/pi/Desktop/MatchingResults"
FPS = 10
RESOLUTION = (640, 480)
REMOTE_SERVER = "www.google.com"
# To draw entrance line
init_point = (0, 200)
end_point = (550, 420)
kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
time_start = time.time()

class MainApp():

    def __init__(self, camera, ts_obj, matcher, codec, extension, db_obj, obf_obj):
        '''
        Application constructor.

        Args:
        -----
            ts_obj (TrackingSystem) : obj for entry/exit detection
            matcher (Matcher) : obj for matching algorithm
            codec (String) : codec to use
            extension (String) : video file extension
            db_obj (Database) : obj to database interaction
            obf_obj (FaceAnonymizer) : obj that provides obfuscation algorithm
        '''
        super(MainApp, self).__init__()

        # create video capture
        self.__camera = camera
        #        self.__camera.start()
        time.sleep(1)

        self.__ts_obj = ts_obj
        self.__matcher = matcher
        self.__event_flag = None
        self.__db_obj = db_obj
        self.__obf_obj = obf_obj
        self.__extension = extension

        # create dirs
        if not os.path.exists(VIDEOS_TEMP_DIR):
            os.makedirs(VIDEOS_TEMP_DIR)

        if not os.path.exists(MATCHING_DIR):
            os.makedirs(MATCHING_DIR)

        # create video writter
        self.__filename = time.strftime("%Y%m%d-%H%M%S")
        self.__video_writer = cv2.VideoWriter(VIDEOS_TEMP_DIR + "/" + self.__filename + "." + self.__extension,
                                              cv2.VideoWriter_fourcc(*codec), FPS, RESOLUTION)
        
        logging.basicConfig(filename="prints.log", level=logging.INFO)
        logging.info("App started")

    def run(self):
        '''
        Main cycle of the application. Check block
        diagrams for more information.
        '''
        #        self.__camera.start()
        # TESTES
        previous_frame= None        

        try:
            time_passed = 0
            while True and time_passed<86400:
                time_passed = time.time() - time_start
                print("App running")
                ret, og_frame = self.__camera.read()
                #                bgr_frame = self.__camera.read()
                if ret == False:
                    print("No frame read")
                    break

                bgr_frame = og_frame.copy()
                
                if previous_frame is not None:
                
                
                    
                    difference_between_frames = cv2.absdiff(previous_frame,bgr_frame)
                    gray_frame = cv2.cvtColor(difference_between_frames, cv2.COLOR_BGR2GRAY)
                    th_frame = cv2.threshold(gray_frame, 50, 255, cv2.THRESH_BINARY)[1]
                    dilated = cv2.dilate(th_frame, kernel)
                    _, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for c in contours:
                        if cv2.contourArea(c)> 50:
                            self.__write_frame(bgr_frame)
                            break
                
                previous_frame = bgr_frame
                    
                # stop condition
                if os.path.isfile('stop'):
                    print("Stopped App")
                    break

                # get needed frames and process
                gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
                #                print("Before")
                self.__ts_obj.process_frame(gray_frame, bgr_frame)
                # write frame if motion is detected

#                if len(self.__ts_obj.last_frame_persons) != 0:
                    #                    print("Persons Detected")
                    # self.__write_frame(bgr_frame)
#                    for person in self.__ts_obj.last_frame_persons:
#                        centroid = person.centroid
#                        cv2.putText(bgr_frame, "ID " + str(person.id), centroid, cv2.FONT_HERSHEY_COMPLEX, 0.5,
#                                    (255, 255, 0), 2)
                #                print("No Persons Detected")

                # draw counting line
                cv2.line(bgr_frame, init_point, end_point, (255, 0, 0), thickness=2)

                # draw text for number of entries and exitsevent_type + " results"
                entries = self.__ts_obj.nr_entries
                exits = self.__ts_obj.nr_exists
                cv2.putText(bgr_frame, "Entries: " + str(entries), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0),
                            2)
                cv2.putText(bgr_frame, "Exits: " + str(exits), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

                #                print("Entries: " + str(entries))
                #                print("Exits: " + str(exits))


                cv2.imshow("frame", bgr_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()
            # apply matching and obfuscation algorithms
            person_storage = self.__ts_obj.person_storage
            matching_results, video_path = self.__apply_algorithms(person_storage)
            ##
            #            # store results
            case_dir = self.__filename
            self.__matcher.write_matches(MATCHING_DIR + "/" + case_dir)
            
            self.__store_data_video(matching_results, video_path)
            

        finally:
            self.__close_resources()

    def __write_frame(self, frame):
        '''
        Writes the frame in the video.

        Args:
        -----
            frame (ndarray) : frame to write
        '''
        self.__video_writer.write(frame)

    def __apply_algorithms(self, person_storage):
        '''
        Applies matching and obfuscation algorithms.

        Args:
        -----
            person_storage (PersonStorage) : obj that stores persons

        Return:
        -------
            (list, String) : matching results and path to obfuscated video
        '''
        # apply matching algorithm
        print("Matching Persons")
        logging.info("Matching Persons")
        self.__matcher.set_person_storage(person_storage)
        logging.info("Setiing Person storage")
        # results = self.__matcher.apply_algorithm()
        
        for person in person_storage.entries:
#                print(person.person_frame.shape)
            person.calc_hists(person.og_frame)
            person.apply_descriptors()
        for person in person_storage.exits:
            person.calc_hists(person.og_frame)
            person.apply_descriptors()
#                print(person.person_frame.shape)
        
        results = self.__matcher.apply_algorithm()
        logging.info("Apply algorithm")

        # apply face anonymizer
        print("Obfuscating faces")
        logging.info("Obfuscating faces")
        path = self.__obf_obj.face_anonymizer(VIDEOS_TEMP_DIR + "/" + self.__filename,
                                              VIDEOS_TEMP_DIR + "/" + self.__filename + "." + self.__extension)

        # delete file without obfuscation from file system
        os.remove(VIDEOS_TEMP_DIR + "/" + self.__filename + "." + self.__extension)
        print("Done")
        logging.info("Done")

        return results, path

    def __store_data_video(self, data, video_path):
        '''
        Stores the extracted data and its video. If there is
        a connection to amazon it is directly stored in amazon services.
        Otherwise its stored localy.

        Args:
        -----
            data (list) : data to store
            video_path (String) : path to the video
        '''

        if self.__check_internet() == True:
            print("Saving on Amazon")
            a = Amazon_API()
            conexao = a.connect_to_rds()
            conexao.insert(data, video_path)
            conexao.close()
            a.connect_to_s3()
            _, s3_filename = os.path.split(video_path)
            a.upload(video_path, s3_filename)
        else:
            print("Saving Locally")
            self.__db_obj.insert(data, video_path)

    def __close_resources(self):
        '''
        Release used resources.
        '''
        #        self.__camera.stop()
        self.__camera.release()
        self.__video_writer.release()

    def __check_internet(self):
        try:
            # see if we can resolve the host name -- tells us if there is
            # a DNS listening
            host = socket.gethostbyname("www.google.com")
            # connect to the host -- tells us if the host is actually
            # reachable
            s = socket.create_connection((host, 80), 2)
            return True
        except:
            pass
        return False
