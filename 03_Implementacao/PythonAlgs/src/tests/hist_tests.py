'''
Module to test histogram methods.
'''

# built-in
import sys
sys.path.append("..")

# libs
import cv2
import matplotlib.pyplot as plt

# own modules
from tracking_system.feature_extraction import FeatureExtraction
from matching_system.my_matching import MyMatching

__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"

# read image
img_matrix = cv2.imread("../../../Dataset/Imgs/test_img.jpg")
img_matrix2 = cv2.imread("../../../Dataset/Imgs/test_img2.jpg")

# calc hue histogram and plot
hist = FeatureExtraction.my_calc_hist(img_matrix)
plt.figure()
plt.plot(hist)

hist2 = FeatureExtraction.my_calc_hist(img_matrix2)
plt.figure()
plt.plot(hist2)
plt.show()

# check match between 2 histograms
match = MyMatching.histogram_match(hist, hist2)
match2 = MyMatching.histogram_match(hist, hist)
