import unittest
import cv2
from matplotlib import image as mpimg

from LaneDetectionCaliberation import CheckCameraCalibration    

print('-----------------------------Testing starts here-----------------------------------')
class LaneDetectionTestCases(unittest.TestCase):
    image_path =  "image/input_image.jpg"
    def test_null_image_value(self):
        image = cv2.imread(self.image_path)

        self.assertIsNotNone(image,'input image not found')
        
    def test_cameracaliberation(self):
    caliberation = CheckCameraCalibration('camera_cal', 9, 6)
    self.assertIsNotNone(caliberation,'Exception should be raised')

    def test_image_equal(self):
        image = mpimg.imread(self.image_path)
        img = mpimg.imread(self.image_path)
        caliberation = CheckCameraCalibration('camera_cal', 9, 6)
        image1 = caliberation.undistort(img)
        print('printing the image')
        diff = cv2.subtract(image,img)
        # print(diff)
        cv2.imshow('Difference',diff)
