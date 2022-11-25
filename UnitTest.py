import unittest
import cv2
from matplotlib import image as mpimg

from LaneDetectionCaliberation import CheckCameraCalibration
from main import DetectLanes
import os

print('-----------------------------Testing starts here-----------------------------------')
class LaneDetectionTestCases(unittest.TestCase):
    image_path =  "image/input_image.jpg"
    def test_null_image_value(self):
        image = cv2.imread(self.image_path)

        self.assertIsNotNone(image,'input image not found')


    def test_cameracaliberation(self):
        caliberation = CheckCameraCalibration('camera_cal', 9, 6)
        self.assertIsNotNone(caliberation,'Exception should be raised')

