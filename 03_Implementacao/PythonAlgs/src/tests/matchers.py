import cv2
import numpy as np

train_samples = np.array([[1.2, 1.0], [1.5, 3.2], [2.2, 9.8]], dtype="float32")
query_samples = np.array([[1.0, 1.0], [1.0, 1.0], [2.2, 9.8]], dtype="float32")

bf = cv2.BFMatcher()
matches = bf.knnMatch(query_samples, train_samples, k=10)
