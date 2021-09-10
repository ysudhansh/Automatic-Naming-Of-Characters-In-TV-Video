import cv2
import numpy as np
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import time
from imutils.video import FileVideoStream
import os

def individual_faces(folder):

    characters = {'Jim':[], 'Pam':[], 'Michael':[], 'Dwight':[], 'Andy':[]}
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor.dat")

    for i in os.listdir(folder):
        #print(i.split('.'))
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
            #cv2.imshow("out", res)
            #cv2.waitKey(1)
            characters[name].append(res)

    return characters


def rmsdiff(im1, im2):
    img = im1 - im2
    h,bins = np.histogram(img.ravel(),256,[0,256])
    sq = (value*(idx**2) for idx, value in enumerate(h))
    sum_of_squares = np.sum(sq)
    rms = np.sqrt(sum_of_squares/float(im1.shape[0] * im1.shape[1]))
    return rms

def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    return z

def face_detect(vid, subs):
    characters = individual_faces("./chars")
    fvs = FileVideoStream(vid).start()
    time.sleep(0.5)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor.dat")
    old_mouth = None
    curr_mouth = None
    frm = 0
    flag = 0
    prev_L = []
    prev_len = 0
    img_array = []
    max_i = []
    for i in range(2675):
        max_i.append(-1)
    #print(subs)
    #loop = 0
    while fvs.more():
        if frm==2674:
            break
        #i = i+1
        frm += 1
        print(frm)
        image = fvs.read()
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 3)
        xbox = []
        for rect in rects:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            xbox.append(x)
        rect_size = len(rects)
        temp = sort_list(rects, xbox)
        rects = temp
        if flag == 0:
            prev_len = len(rects)
            prev = list(range(len(rects)))
            curr = list(range(len(rects)))
            dist = list(range(len(rects)))
        else:
            if prev_len != len(rects):
                flag = 0
                prev = list(range(len(rects)))
                curr = list(range(len(rects)))
                dist = list(range(len(rects)))
                prev_len = len(rects)

        if len(rects) == 0:
            continue
        else:
            for (i, rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                clone = image.copy()
                for (x,y) in shape[48:68]:
                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                    (x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]]))
                    roi = image[y:y + h, x:x + w]
                    roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
                        
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                res = cv2.resize(roi_gray,(100,250),interpolation = cv2.INTER_CUBIC)

                # Uncomment the below 2 lines for the mouth thing
                # cv2.imshow("lips", res)
                # cv2.waitKey(1)
                
                if flag == i:
                    prev[i] = res
                    curr[i] = res
                    flag += 1
                else:
                    curr[i] = res
                    dist[i] = rmsdiff(curr[i], prev[i])
                    prev[i] = curr[i]

            curr_char = []
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
                min_char = 1e8
                min_char_name = None
                for char in characters.keys():
                    for chari in characters[char]:
                        dist_char = rmsdiff(res, chari)
                        if dist_char < min_char:
                            min_char = dist_char
                            min_char_name = char
                curr_char.append(min_char_name)
            
            max_dist_char = np.argmax(np.array(dist))
            max_i[frm] = max_dist_char
            # print("max: ", max_dist_char)
            flagname = 0
            flagnone = 0
            name = subs[frm]
            if name == None:
                flagnone = 1
            # print(flagnone)
            if name == subs[frm - 1]:
                flagname = 1
            usei = max_i[frm - 1]
            # print("usei: ", usei)
            for (i, rect) in enumerate(rects):
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if flagnone == 1:
                    
                    cv2.putText(image, "{}: Non-Speaker".format(curr_char[i]), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # print("flag: ", flagname)    
                    if i == max_dist_char:
                        if flagname == 0:
                            cv2.putText(image, "{}: Speaker".format(name), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        if flagname == 1:
                            if i == usei:
                                cv2.putText(image, "{}: Speaker".format(name), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                max_i[frm] = usei
                            else:
                                cv2.putText(image, "{}: Non-Speaker".format(curr_char[i]), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        if flagname == 0:
                            cv2.putText(image, "{}: Non-Speaker".format(curr_char[i]), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        if flagname == 1:
                            if i == usei:
                                cv2.putText(image, "{}: Speaker".format(name), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                max_i[frm] = usei
                            else:
                                cv2.putText(image, "{}: Non-Speaker".format(curr_char[i]), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        img_array.append(image)
        height, width, layers = image.shape
        size = (width,height)
        cv2.imshow("Output", image)
        cv2.waitKey(1)
        

    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    print(len(img_array))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


