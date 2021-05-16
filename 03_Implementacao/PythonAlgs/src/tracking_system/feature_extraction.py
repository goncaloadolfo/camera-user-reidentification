'''
Person features extraction module.
'''

# libs
import cv2
import numpy as np
from libs.calcCSD import compCSD
from libs.calcEHD import compEHD

__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"
MIN_SHAPE = 272


class FeatureExtraction:

    @staticmethod
    def my_calc_hist(frame, channel=[0], mask=None, bins=[180], ranges=[0, 180]):
        '''
        Calculates 1D histogram with hsv space color counting.
        Default parameters set for hue plan.

        Args:
        -----
            frame (ndarray) : BGR frame
            channel (int) : defines which plan to use, default 0(hue)
            mask (ndarray) : mask to apply counting process, default None
            bins (int) : number of bins, default 180
            ranges (list) : range of values, default [0, 180] (hue)
        '''
        # print("frame shape: ", frame.shape)
        # convert to hsv space
        hsv_matrix = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # print(hsv_matrix)

        # calc histogram
        hist = cv2.calcHist([hsv_matrix], channel, mask, bins, ranges)

        # get number of pixels
        nr_pixels = np.sum(hist)
        # print("pixels count: ", nr_pixels)

        # normalize
        normalized_hist = hist * 1.0 / nr_pixels

        return normalized_hist

    @staticmethod
    def csd(img, nr_colors=64):
        '''
        Apply color structured descriptor. Returns

        Args:
        -----
            img (ndarray) : bgr img numpy array
            nr_colors (int) : number of colors to be considered(32, 64, 128, 256), default 64

        Return:
        -------
            (ndarray) : array of shape (nr_colors,)
        '''
        return compCSD(img, nr_colors)

    @staticmethod
    def ehd(img):
        '''
        Applies edge histogram descriptor. Divides gray image into 16 macroblocks. At each macroblock, will be
        extracted quantified edges information(between 0 and 7):
        [vertical, horizontal, diag-45, diag-135, non-directional]

        Args:
        -----
            img (ndarray) : bgr img numpy array

        Return:
        ------
            (ndarray) : array of shape (16*5=80)
        '''
        # get gray frame
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resize if shape is lower than 272x272
        min_shape = min(gray_img.shape)
        aux_img = gray_img

        if min_shape < MIN_SHAPE:
            scale = MIN_SHAPE * 1.0 / min_shape * 1.0
            height, width = gray_img.shape
            aux_img = cv2.resize(aux_img, (int(height * scale), int(width * scale)))

        # apply edge histogram descriptor
        return compEHD(aux_img)
