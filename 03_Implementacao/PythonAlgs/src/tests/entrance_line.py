import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
pts = []
drawing = False
# load the image, clone it, and setup the mouse callback function
video_capture = cv2.VideoCapture("../../Dataset/videos/example_02-sm.mp4")
_, image = video_capture.read()
video_capture.release()
clone = image.copy()
cv2.namedWindow("image")


def click_and_crop(event, x, y, flags, param):
    global pts, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        pts = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:
        pts.append((x, y))
        drawing = False

        cv2.line(image, pts[0], pts[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


def get_line(frame=image):
    cv2.setMouseCallback("image", click_and_crop)
    while True:
        cv2.imshow("image", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            frame = clone.copy()

        elif key == ord("q"):
            cv2.destroyAllWindows()
            break

    return pts


