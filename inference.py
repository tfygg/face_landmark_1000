import torch
import numpy as np
import cv2
import time


class FaceLandmark(object):
    def __init__(self, model):
        self.model = torch.load(model, map_location=torch.device('cpu'))
        self.model.eval()
        self.image_size = 128

    def predict(self, image):
        processed_image = self.process(image)
        result = self.model(processed_image)
        result = result.detach().numpy()
        landmark = np.array(result[0][0:136]).reshape([-1, 2])
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


def vis(model, image):
    image = cv2.resize(image, (128, 128))
    img_show = image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image)
    face = torch.load(model, map_location=torch.device('cpu'))
    face.eval()
    start=time.time()
    res = face(image)
    res = res.detach().numpy()
    print(res)
    print('xxxx', time.time() - start)
    landmark = np.array(res[0][0:136]).reshape([-1, 2])
    print(landmark)
    print(landmark*128)
    for _index in range(landmark.shape[0]):
        x_y = landmark[_index]
        # print(x_y)
        cv2.circle(img_show, center=(int(x_y[0] * 128),
                                     int(x_y[1] * 128)),
                   color=(255, 122, 122), radius=1, thickness=2)
    img_show = cv2.resize(img_show, (420, 420))
    cv2.imshow('tmp', img_show)
    cv2.waitKey(0)


if __name__ == '__main__':
    model = './model/keypoints.pth'
    image = cv2.imread('./model/angelababy.jpg')
    handle = FaceLandmark(model)
    handle.predict(image)
    # vis(model, image)
