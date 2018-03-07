from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

data, labels = [], []

dir_path = './dataset/train'
imagePaths = list(paths.list_images(dir_path))

rand_spot = random.randint(0, len(imagePaths)-10)
imagePaths = imagePaths[rand_spot:rand_spot+10]

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    features = image_to_feature_vector(image)
    data.append(features)
    labels.append(label)

le = LabelEncoder()
labels = le.fit_transform(labels)
data = np.array(data) / 255.0
labels = np_utils.to_categorical(labels, 2)
testData = data

# --------------------------------------------------------------
from keras.models import load_model

model_name = "catsdogs_model.h5"
model = load_model(model_name)

# (trainData, testData, trainLabels, testLabels) = train_test_split(
# 	data, labels, test_size=0.9, random_state=42)
# (loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
# print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

for i, data in enumerate(testData):
    img = plt.imread(imagePaths[i])
    prob = model.predict(data.reshape(1,data.shape[0]))
    plt.imshow(img)
    label = labels[i]
    label_predicted = "cat" if prob[0][0]>prob[0][1] else "dog"
    label_actual = "CAT" if label[0]>label[1] else "DOG"
    title_l = label_predicted + "<-- Prediction\n" + label_actual + "<-- Actual"
    plt.title(title_l)
    plt.show()
