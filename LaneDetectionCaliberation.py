import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

# Caliberate the image with the chess board images already captured for particular camera
class CheckCameraCalibration():
    def __init__(self, image_dir, nx, ny, debug=False):
        fnames = glob.glob("{}/*".format(image_dir))
        objpoints = []
        imgpoints = []

        objp = np.zeros((nx * ny, 3), np.float32)
        print(objp)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        print(objp)
        for f in fnames:
            img = mpimg.imread(f)
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(img, (nx, ny))
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)

        shape = (img.shape[1], img.shape[0])
        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

        if not ret:
            raise Exception("Unable to calibrate camera")

    def returnCheckCameraCalibrationdata(self):
        return self.mtx,self.dist

    def undistort(self, img):

        plt.imshow(cv2.undistort(img, self.mtx, self.dist, None, self.mtx))
        plt.show()
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
