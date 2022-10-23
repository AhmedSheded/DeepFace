import os
import random
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split


# def preprocess_img(img):
#     img /= 255.0
#     return img

def normalization(img):

    for i in range(3):
        img[:,:,i]=np.absolute((img[:,:,i]-np.min(img[:,:,i]))/(np.max(img[:,:,i])-np.min(img[:,:,i])))*255
    return img


def read_images(path, image_size, num_samples):
    names=[]
    training_images, training_labels = [], []
    label = 0
    for dirname, subdirnames, filenames in os.walk(path):
        for subdirname in subdirnames:
            names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            samples = random.sample(os.listdir(subject_path), num_samples)
            for filename in samples:
                img = normalization(cv.imread(os.path.join(subject_path, filename)))
                if img is None:
                    continue
                img = cv.resize(img, image_size)
                training_images.append(img)
                training_labels.append(label)
            label+=1
    training_images = np.asarray(training_images, np.uint8)
    training_labels = np.asarray(training_labels, np.int32)
    trainX, valX, trainy, valy = train_test_split(training_images, training_labels, random_state=222, test_size=0.2)
    return names, trainX, trainy, valX, valy
