import cv2
import numpy as np
# from obfuscation_system.face import Face
import time
from PIL import Image
from edgetpu.detection.engine import DetectionEngine


class FaceAnonymizer:

    def __init__(self, face_engine, person_engine, threshold=0.3):
        self.__face_engine = face_engine
        self.__person_engine = person_engine
        self.__threshold = threshold
        self.__saved_video = "event_"
        self.__event_counter = 0
        self.__total_frames = 0
        self.__faces = []

    def __detector(self, bgr_frame, engine):
        '''
        Detects faces on the frame
        :param 2D uint8 numpy array frame
        :return 4D uint8 numpy array: exits counter value
        '''
        frame = Image.fromarray(bgr_frame)

        # Run Inference
        # threshold - float defining the minimun confidence threshold
        # top_k - int defining maximum number of objects to detect
        # relative_coord - bool defining if returns float coords or int
        results = engine.DetectWithImage(frame, threshold=self.__threshold, keep_aspect_ratio=True,
                                         relative_coord=False, top_k=10)

        return results

    def face_anonymizer(self, output_name, video=0):
        '''
        Show blurred faces on frame
        '''
        cap = cv2.VideoCapture(video)
        _, frame = cap.read()

        out, out_name = self.__create_video(output_name)

        while True:
            _, frame = cap.read()

            if frame is None:
                break

            self._persons = []
            detected_persons = self.__detector(frame, self.__person_engine)

            if detected_persons:
                for obj in detected_persons:
                    if obj.label_id == 0:
                        bounding_box = obj.bounding_box.flatten().tolist()
                        a_x = int(bounding_box[0])
                        a_y = int(bounding_box[1])
                        b_x = int(bounding_box[2])
                        b_y = int(bounding_box[3])
                        person_frame = frame[a_y: b_y, a_x: b_x].copy()

                        #                        cv2.imshow("Person",person_frame)

                        detected_face = self.__detector(person_frame, self.__face_engine)
                        if len(detected_face) > 0:

                            face_bounding_box = detected_face[0].bounding_box.flatten().tolist()
                            face_a_x = int(face_bounding_box[0])
                            face_a_y = int(face_bounding_box[1])
                            face_b_x = int(face_bounding_box[2])
                            face_b_y = int(face_bounding_box[3])

                            #                            face_frame = person_frame[face_a_y: face_b_y, face_a_x: face_b_x].copy()
                            #                            cv2.imshow("FACE",face_frame)

                            frame = self.face_blur(frame, face_a_x + a_x, a_y + face_a_y, a_x + face_b_x,
                                                   face_b_y + a_y)



                        else:

                            # Obfuscation trough persons detected
                            cy = int((a_y + b_y) / 2)
                            frame = self.face_blur(frame, a_x, a_y, b_x, cy)

            cv2.imshow("Face Blur", frame)

            out.write(frame)
            self.__total_frames += 1

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return out_name

    def face_blur(self, frame, upper_left_x, upper_left_y, lower_right_x, lower_right_y):
        '''
        Blurs box of detected face
        :param 2D uint8 numpy array frame: current frame
        :param int upper_left_x: upper left x coordinate of bounding box
        :param int upper_left_y: upper left y coordinate of bounding box
        :param int lower_right_x: lower right x coordinate of bounding box
        :param int lower_right_y: lower right y coordinate of bounding box
        :return 2D uint8 numpy array frame: frame with blurred face
        '''
        frame[upper_left_y: lower_right_y, upper_left_x: lower_right_x] = cv2.blur(
            frame[upper_left_y: lower_right_y, upper_left_x: lower_right_x],
            (30, 30))
        return frame

    def __create_video(self, output_name):
        '''
        Creates VideoWriter object to write frames into.
    
        Args:
        -----
    
    
        Return:
        -------
            (VideoWriter) : output to write to
        '''
        # Define the codec and create VideoWriter object
        #        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #        output_name += "_obf.avi"
#        fourcc = cv2.VideoWriter_fourcc('V', 'P', '8', '0')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        output_name += "_obf.mp4"

        out = cv2.VideoWriter(output_name, 0x00000021, 10.0, (640, 480))
        return out, output_name


if __name__ == "__main__":
    print("Obfuscating...")
    # obfuscation system
    face_engine = DetectionEngine("../../models/edge_tpu_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite")
    person_engine = DetectionEngine(
        "../../models/edge_tpu_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")
    fa = FaceAnonymizer(face_engine, person_engine)
    video_path = fa.face_anonymizer("output", "../../../Dataset/videos/VideoTest3.avi")
