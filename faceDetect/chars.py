import os
import cv2
import numpy as np
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import time

def individual_faces(folder):

    characters = {'Jim':[], 'Pam':[], 'Michael':[], 'Dwight':[], 'Andy':[]}
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor.dat")

    for i in os.listdir(folder):
        print(i.split('.'))
        name = i.split('.')[0][:-1]
        name = name.title()
        image = cv2.imread(folder + "/" + i)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            clone = image.copy()
            for (x,y) in shape:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                (x, y, w, h) = cv2.boundingRect(np.array([shape]))
                roi = image[y:y + h, x:x + w]
                roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            res = cv2.resize(roi_gray,(250,250),interpolation = cv2.INTER_CUBIC)
            cv2.imshow("out", res)
            cv2.waitKey(1)
            characters[name].append(res)

    return characters

chars = individual_faces(os.path.abspath("./chars"))
print(chars)