# built-in

# libs
import cv2
import numpy as np

# own libs


def write_frames(dir_path, video_path):
    video_capture = cv2.VideoCapture(video_path)
    frame_counter = 0

    while True:
        ret, frame = video_capture.read()

        if frame is None:
            break

        cv2.imwrite(dir_path + "/frame" + str(frame_counter) + ".tif", frame)
        frame_counter += 1

    video_capture.release()


def threshold_impact(frame1, frame2, thr, morph=False):
    frame1_matrix = cv2.imread(frame1)
    frame2_matrix = cv2.imread(frame2)

    frame1_gray = cv2.cvtColor(frame1_matrix, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2_matrix, cv2.COLOR_BGR2GRAY)

    dif_frame = np.abs(frame1_gray * 1.0 - frame2_gray * 1.0)
    binary_frame = cv2.threshold(dif_frame, thr, 255.0, cv2.THRESH_BINARY)[1]

    if morph:
        dilate_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        binary_frame = cv2.dilate(binary_frame.astype(np.uint8), dilate_elem, iterations=3)

    cv2.imwrite("motionPixels" + str(thr) + ".tif", binary_frame.astype(np.uint8))


if __name__ == "__main__":
    # write_frames("frames", "../../../Dataset/videos/VideoTest1.avi")
    threshold_impact("frames/frame558.tif", "frames/frame559.tif", 40, True)
    # threshold_impact("frames/frame559.tif", "C:/Sem6/Projeto/Dataset/Imgs/BackgroundImg2.tif", 40)
    print("")
