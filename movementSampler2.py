#!/usr/bin/env python3.5

import argparse
import logging
import time

import cv2
import numpy as np
import tensorflow as tf
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import datetime

from threading import Thread
import pickle
import io


# This file(run_webcam.py) is heavily changed to capture
# and store standard pose data for further machine learning
class WebcamVideoStream:
    def __init__(self, capture):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = capture
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


# define binery data
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# define integer data
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# define float data
def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def textCountdown(t, mString):
    progressBar = '>>>>>>>>>'
    while t:
        print(progressBar + mString + str(t))
        time.sleep(1)
        t -= 1
        progressBar += '>>>>>>>>>'

def captureImage(recordTypeNum, recording_time,recording_interval, vs, imageList):
    recordTypes = ['000 GOOD MOVES 000','111 BAD MOVES 111']
    recordTypeString = recordTypes[recordTypeNum]
    textCountdown(3, 'Capture session' + recordTypeString +' start in :' )
    start_time = time.time()

    while True:
        if ((time.time() - start_time) >= recording_time):
            print('Time up!')
            break
        else:
            image = vs.read()

            group_index =int(np.floor((time.time() - start_time)/recording_interval))
            print('adding image to group:' + str(group_index) )
            imageList[group_index] += [image]
            # cv2.putText(image,str(len(HumanFrames)),(10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            image = None
            if cv2.waitKey(1) == 27:
                break
            cv2.waitKey(10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='304x224',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)

    print("WebcamVideoStream instance 'vs' is created")
    vs = WebcamVideoStream(cam).start()

    #frameNumber = 0
    goodSampleSize = 5
    badSampleSize = 5

    recording_time = 4.0
    recording_interval = 0.2
    groupNumber = np.floor(recording_time / recording_interval)

    imageSetGood = []
    imageSetBad = []
    dataSet = []
    exerciseName = 'torsoTwist'
    for i in range(int(groupNumber)):
        imageSetGood.append([])
    for i in range(int(groupNumber)):
        imageSetBad.append([])
    for i in range(int(groupNumber)):
        dataSet.append([])
    # label for good moves:0 / bad moves: 1
    labelSet=[]
    for i in range(int(groupNumber)):
        labelSet.append([])
    body = []
    timeRatio = []



    #process time of pose estimation with current setting is about 0.125 sec/frame
    #frameLimit = int(round(recording_time/0.125))
    #print('Limit of frame number is:' + str(frameLimit))

    #Target exercises:In situ high knee / wood chop / Plank*40Sec


    print(exerciseName +', left *1 right *1 ,recording time=' + str(recording_time) + 'Secs')
    for i in range(goodSampleSize):
        captureImage(0,recording_time,recording_interval,vs,imageSetGood)

    for i in range(badSampleSize):
        captureImage(1, recording_time, recording_interval, vs, imageSetBad)


    for i in range(int(groupNumber)):
        print('processing Good Sample of group number:' +str(i))
        for image in imageSetGood[i]:
            temp = []
            imageC = image.copy()
            humans = e.inference(imageC, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            #print(humans)
            if ((humans != []) & (len(humans)==1)):
                for human in humans:
                    for j in range(common.CocoPart.Background.value):
                        if j not in human.body_parts.keys():
                            temp.append(0.0)
                            temp.append(0.0)
                            continue
                        body_part = human.body_parts[j]
                        coord = [body_part.x, body_part.y]
                        temp.append(coord[0])
                        temp.append(coord[1])

                dataSet[i].append(temp)

                labelSet[i].append(0)
            #imageC = TfPoseEstimator.draw_humans(imageC, humans, imgcopy=False)
            #cv2.imshow("process result", imageC)
            imageC = None
            #cv2.waitKey(5)

    for i in range(int(groupNumber)):
        print('processing Bad sample of group number:' + str(i))
        for image in imageSetBad[i]:
            temp = []
            imageC = image.copy()
            humans = e.inference(imageC, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            #print(humans)
            if ((humans != []) & (len(humans) == 1)):
                for human in humans:
                    for j in range(common.CocoPart.Background.value):
                        if j not in human.body_parts.keys():
                            temp.append(0.0)
                            temp.append(0.0)
                            continue
                        body_part = human.body_parts[j]
                        coord = [body_part.x, body_part.y]
                        temp.append(coord[0])
                        temp.append(coord[1])

                dataSet[i].append(temp)

                labelSet[i].append(1)
            #imageC = TfPoseEstimator.draw_humans(imageC, humans, imgcopy=False)
            #cv2.imshow("process result", imageC)
            imageC = None
            #cv2.waitKey(5)

    #Free memory space for better processing speed
    del imageSetGood
    del imageSetBad
    print('image set flushed!!')
    #Restart imageSets
    imageSetGood = []
    imageSetBad = []
    for i in range(int(groupNumber)):
        imageSetGood.append([])
    for i in range(int(groupNumber)):
        imageSetBad.append([])

    print(exerciseName+', left *1 right *1 ,recording time=' + str(recording_time) + 'Secs')
    for i in range(goodSampleSize):
        captureImage(0, recording_time, recording_interval, vs, imageSetGood)

    for i in range(badSampleSize):
        captureImage(1, recording_time, recording_interval, vs, imageSetBad)

    for i in range(int(groupNumber)):
        print('processing Good Sample of group number:' + str(i))
        for image in imageSetGood[i]:
            temp = []
            imageC = image.copy()
            humans = e.inference(imageC, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            #print(humans)
            if ((humans != []) & (len(humans) == 1)):
                for human in humans:
                    for j in range(common.CocoPart.Background.value):
                        if j not in human.body_parts.keys():
                            temp.append(0.0)
                            temp.append(0.0)
                            continue
                        body_part = human.body_parts[j]
                        coord = [body_part.x, body_part.y]
                        temp.append(coord[0])
                        temp.append(coord[1])

                dataSet[i].append(temp)

                labelSet[i].append(0)
            # imageC = TfPoseEstimator.draw_humans(imageC, humans, imgcopy=False)
            # cv2.imshow("process result", imageC)
            imageC = None
            # cv2.waitKey(5)

    for i in range(int(groupNumber)):
        print('processing Bad sample of group number:' + str(i))
        for image in imageSetBad[i]:
            temp = []
            imageC = image.copy()
            humans = e.inference(imageC, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            #print(humans)
            if ((humans != []) & (len(humans) == 1)):
                for human in humans:
                    for j in range(common.CocoPart.Background.value):
                        if j not in human.body_parts.keys():
                            temp.append(0.0)
                            temp.append(0.0)
                            continue
                        body_part = human.body_parts[j]
                        coord = [body_part.x, body_part.y]
                        temp.append(coord[0])
                        temp.append(coord[1])

                dataSet[i].append(temp)

                labelSet[i].append(1)
            # imageC = TfPoseEstimator.draw_humans(imageC, humans, imgcopy=False)
            # cv2.imshow("process result", imageC)
            imageC = None
            # cv2.waitKey(5)

    # Free memory space for better processing speed
    del imageSetGood
    del imageSetBad
    print('image set flushed!!')
    # Restart imageSets
    imageSetGood = []
    imageSetBad = []
    for i in range(int(groupNumber)):
        imageSetGood.append([])
    for i in range(int(groupNumber)):
        imageSetBad.append([])

    print(exerciseName+', left *1 right *1 ,recording time=' + str(recording_time) + 'Secs')
    for i in range(goodSampleSize):
        captureImage(0, recording_time, recording_interval, vs, imageSetGood)

    for i in range(badSampleSize):
        captureImage(1, recording_time, recording_interval, vs, imageSetBad)

    for i in range(int(groupNumber)):
        print('processing Good Sample of group number:' + str(i))
        for image in imageSetGood[i]:
            temp = []
            imageC = image.copy()
            humans = e.inference(imageC, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            # print(humans)
            if ((humans != []) & (len(humans) == 1)):
                for human in humans:
                    for j in range(common.CocoPart.Background.value):
                        if j not in human.body_parts.keys():
                            temp.append(0.0)
                            temp.append(0.0)
                            continue
                        body_part = human.body_parts[j]
                        coord = [body_part.x, body_part.y]
                        temp.append(coord[0])
                        temp.append(coord[1])

                dataSet[i].append(temp)

                labelSet[i].append(0)
            # imageC = TfPoseEstimator.draw_humans(imageC, humans, imgcopy=False)
            # cv2.imshow("process result", imageC)
            imageC = None
            # cv2.waitKey(5)

    for i in range(int(groupNumber)):
        print('processing Bad sample of group number:' + str(i))
        for image in imageSetBad[i]:
            temp = []
            imageC = image.copy()
            humans = e.inference(imageC, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            # print(humans)
            if ((humans != []) & (len(humans) == 1)):
                for human in humans:
                    for j in range(common.CocoPart.Background.value):
                        if j not in human.body_parts.keys():
                            temp.append(0.0)
                            temp.append(0.0)
                            continue
                        body_part = human.body_parts[j]
                        coord = [body_part.x, body_part.y]
                        temp.append(coord[0])
                        temp.append(coord[1])

                dataSet[i].append(temp)

                labelSet[i].append(1)
            # imageC = TfPoseEstimator.draw_humans(imageC, humans, imgcopy=False)
            # cv2.imshow("process result", imageC)
            imageC = None
            # cv2.waitKey(5)
    print(dataSet)
    print(labelSet)
    print(np.shape(dataSet))
    print(np.shape(labelSet))

    output = open( exerciseName+'Data3.pkl', 'wb')
    pickle.dump(dataSet, output)
    output.close()

    output = open(exerciseName+'Label3.pkl', 'wb')
    pickle.dump(labelSet, output)
    output.close()

    pkl_file = open(exerciseName+'Data3.pkl', 'rb')
    highKneeData = pickle.load(pkl_file)
    pkl_file.close()
    print(highKneeData)

    pkl_file = open(exerciseName + 'Label3.pkl', 'rb')
    highKneeLabel = pickle.load(pkl_file)
    pkl_file.close()
    print(highKneeLabel)

cv2.destroyAllWindows()
vs.stop()
