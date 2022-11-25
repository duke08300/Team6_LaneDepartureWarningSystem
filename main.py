from LaneDetectionCaliberation import *

class DetectLanes:

    def __init__(self):
        self.calibration = CheckCameraCalibration('camera_cal', 9, 6)

        self.thresholding = Threshold()
        self.transform = PerspectiveTransformation()
        self.lanedirection = DetectLaneDirection()
    def forward(self,img):
        #Copy image path
        out_img = np.copy(img)

        #Converting to grey
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanedirection.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanedirection.plot(out_img)

        return out_img

    def process_image(self, input_path):
      #Read the path of the image
        img = mpimg.imread(input_path)

        out_img = self.forward(img)

        plt.imshow(out_img)
        plt.show()


def main():
    findlanes = DetectLanes()
    image_path = "image/input_image3.jpg"
    findlanes.process_image(image_path)


if __name__ == "__main__":
    main()

