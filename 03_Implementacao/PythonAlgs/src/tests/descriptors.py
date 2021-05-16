import cv2
from libs.calcCSD import compCSD

img = cv2.imread("../../../Dataset/Imgs/test_img.jpg")
csd = compCSD(img, 64)
