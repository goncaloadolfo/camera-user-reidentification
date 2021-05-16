import cv2
import numpy as np



class FaceAnonymizer:

    def __init__(self, face_engine, person_engine, threshold=0.3):
        self.__face_engine = face_engine
        self.__person_engine = person_engine
        self.__threshold = threshold
        self.__saved_video = "event_"
        self.__event_counter = 0
        self.__total_frames = 0
        self.__faces = []

    def __face_detector(self, bgr_frame, net):
        '''
        Detects faces on the frame
        :param 2D uint8 numpy array frame
        :return 4D uint8 numpy array: exits counter value
        '''
        # executa subtracao de media, e escala imagem
        blob = cv2.dnn.blobFromImage(cv2.resize(bgr_frame, (300, 300)), 1.0,
                                     # blob - coleccao de imagens com as mesmas dimensoes
                                     (300, 300), (104.0, 177.0,
                                                  123.0))  # argumentos - frame, scale, cnn spatialsize, media R-G-B
        net.setInput(blob)
        faces = net.forward()  # forward pass - proceso de calculo input para output layers
        return faces

    def __person_detector(self, bgr_frame, net):
        '''
        Detects faces on the frame
        :param 2D uint8 numpy array frame
        :return 4D uint8 numpy array: exits counter value
        '''
        # executa subtracao de media, e escala imagem
        frame_resize = cv2.resize(bgr_frame, (300, 300))
        blob = cv2.dnn.blobFromImage(frame_resize, 0.007843,
                                     # blob - coleccao de imagens com as mesmas dimensoes
                                     (300, 300), (127.5, 127.5, 127.5),
                                     False)  # argumentos - frame, scale, cnn spatialsize, media R-G-B
        net.setInput(blob)
        persons = net.forward()  # forward pass - proceso de calculo input para output layers
        return persons

    def face_anonymizer(self, output_name, video=0):
        '''
        Show blurred faces on frame
        '''
        cap = cv2.VideoCapture(video)
        _, frame = cap.read()

        out, out_name = self.__create_video(output_name)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            self._persons = []
            detected_persons = self.__person_detector(frame, self.__person_engine)

            h = frame.shape[:2][0]
            w = frame.shape[:2][1]

            frame_resize = cv2.resize(frame, (300, 300))
            frame_shape = frame_resize.shape[:2]
            resized_h = frame_shape[0]
            resized_w = frame_shape[1]

            for i in range(detected_persons.shape[2]):

                # probabilidade de ser face
                probability = detected_persons[0, 0, i, 2]

                if probability > self.__threshold:
                    id_class_person = int(detected_persons[0, 0, i, 1])  # Class label

                    if id_class_person == 15:  # label 15 refers to class person
                        bounding_box = detected_persons[0, 0, i, 3:7] * np.array(
                            [resized_w, resized_h, resized_w, resized_h])

                        (a_x, a_y, b_x, b_y) = bounding_box.astype("int")

                        # Scaling to original frame size
                        h_factor = frame.shape[0] / 300
                        w_factor = frame.shape[1] / 300

                        # Scaling bounding box to current frame
                        a_x = int(w_factor * a_x)
                        a_y = int(h_factor * a_y)
                        b_x = int(w_factor * b_x)
                        b_y = int(h_factor * b_y)

                        # limit coords
                        a_x = a_x if a_x > 0 else 0
                        a_y = a_y if a_y > 0 else 0
                        a_x = a_x if a_x < frame.shape[1] else frame.shape[1]
                        a_y = a_y if a_y < frame.shape[0] else frame.shape[0]

                        b_x = b_x if b_x < frame.shape[1] else frame.shape[1]
                        b_y = b_y if b_y < frame.shape[0] else frame.shape[0]

                        person_frame = frame[a_y: b_y, a_x: b_x].copy()
                        person_frame_h = person_frame.shape[:2][0]
                        person_frame_w = person_frame.shape[:2][1]

                        #                        cv2.imshow("Person",person_frame)

                        detected_faces = self.__face_detector(person_frame, self.__face_engine)

                        faces = []
                        for i in range(detected_faces.shape[2]):

                            # probabilidade de ser face
                            probability = detected_faces[0, 0, i, 2]

                            if probability > 0.4:
                                bounding_box = detected_faces[0, 0, i, 3:7] * np.array(
                                    [person_frame_w, person_frame_h, person_frame_w, person_frame_h])

                                (face_a_x, face_a_y, face_b_x, face_b_y) = bounding_box.astype("int")
                                if face_a_x > w or face_a_y > h or face_b_x > w or face_b_y > h:
                                    frame = self.face_blur(frame, 0, 0, w, h)
                                    break

                                face_a_x = np.abs(face_a_x)
                                face_a_y = np.abs(face_a_y)
                                face_b_x = np.abs(face_b_x)
                                face_b_y = np.abs(face_b_y)

                                faces.append((face_a_x, face_a_y, face_b_x, face_b_y))

                        if len(faces) > 0:
                            # for face_box in faces:
                            #     print(face_box)
                            #     frame = self.face_blur(frame, face_box[0] + a_x, face_box[1] + a_y, face_box[2] + a_x,
                            #                            face_box[3] + a_y)
                            frame = self.face_blur(frame, faces[0][0] + a_x, faces[0][1] + a_y, faces[0][2] + a_x,
                                                   faces[0][3] + a_y)

                        else:

                            # Obfuscation trough persons detected
                            cy = int((a_y + b_y) / 2)
                            frame = self.face_blur(frame, a_x, a_y, b_x, cy)

            # cv2.imshow("Face Blur", frame)

            out.write(frame)

            self.__total_frames += 1

            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break

        cap.release()
        out.release()
        # cv2.destroyAllWindows()
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
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        output_name += "_obf.mp4"

        out = cv2.VideoWriter(output_name, fourcc, 10.0, (640, 480))
        return out, output_name


if __name__ == "__main__":
    print("Obfuscating...")
    # obfuscation system
    face_net = cv2.dnn.readNetFromCaffe("../../models/deploy.prototxt",
                                        "../../models/res10_300x300_ssd_iter_140000.caffemodel")

    person_net = cv2.dnn.readNetFromCaffe("../../models/MobileNetSSD_deploy.prototxt",
                                          "../../models/MobileNetSSD_deploy.caffemodel")

    fa = FaceAnonymizer(face_net, person_net)
    video_path = fa.face_anonymizer("../../../Dataset/videos/VideoTest1", "../../../Dataset/videos/video-1560953495.mp4")
    # video_path = fa.face_anonymizer("output")
