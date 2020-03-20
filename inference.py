import torch
import numpy as np
import cv2
import time


class FaceLandmark(object):
    def __init__(self, model):
        self.model = torch.load(model, map_location=torch.device('cpu'))
        self.model.eval()
        self.image_size = 128
        self.points_num = 1000

    def predict(self, image):
        processed_image = self.process(image)
        result = self.model(processed_image)
        result = result.detach().numpy()
        landmark = np.array(result[0][0:self.points_num]).reshape([-1, 2])
        landmark = landmark * self.image_size
        self.show_result(image, landmark)

    def process(self, image):
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        return image

    def show_result(self, image, landmark):
        image = cv2.resize(image, (self.image_size, self.image_size))
        for point in landmark:
            cv2.circle(image, center=(int(point[0]), int(point[1])), color=(255, 122, 122), radius=1, thickness=2)
        # image = cv2.resize(image, (420, 420))
        cv2.imshow('tmp', image)
        cv2.waitKey(0)
        

if __name__ == '__main__':
    model = './model/keypoints.pth'
    image = cv2.imread('./model/face.jpg')
    landmark_handle = FaceLandmark(model)
    landmark_handle.predict(image)
