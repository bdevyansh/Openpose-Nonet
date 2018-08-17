#!/usr/bin/env python3.5
import argparse
import logging
import time

import cv2
from threading import Thread

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import math
import numpy as np
import random
import tensorflow as tf
import os
#supress low level warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


def textCountdown(t,mString):
    progressBar = '>>>>>>>>>'
    while t:
        print(progressBar+mString+ str(t))
        time.sleep(1)
        t -= 1
        progressBar += '>>>>>>>>>'

def movementVerefication(groupNumber,prefix,body,images):
    performance = 0
    for i in range(groupNumber):
        tf.reset_default_graph()
        with tf.Session() as sess:
            ##################################################
            # load model #
            save_path = "./models/movement/"+ prefix + str(i) +".meta"
            saver = tf.train.import_meta_graph(save_path)

            #resotre the graph
            saver.restore(sess, "./models/movement/" + prefix + str(i))

            #restore values in the model
            x = tf.get_collection("input")[0]
            y = tf.get_collection("output")[0]
            for j in range(len(body[i])):
                input = [body[i][j]]
                result = sess.run(y, feed_dict={x:input })
                if (result == 0):
                    cv2.putText(images[i][j], 'O  Correct', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 8)
                    performance +=1
                else:
                    cv2.putText(images[i][j], 'X  Wrong!', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 8)

                cv2.imshow('Stored image',images[i][j])
                cv2.waitKey(300)
        sess.close()
    if(performance > int(groupNumber *0.9)):
        print('90~100% precision: You performed well. Good job!!')
    elif(performance > int(groupNumber *0.75)):
        print('75-90% precision: Your moves are acceptable')
    elif(performance < int(groupNumber *0.3)):
        print ('??? WHAT ARE YOU DOING ???')
    else:
        print ('Unacceptable, try do better next time!')
        ##################################################
def exerciseCaputre(vs):
    recording_time = 4
    recording_interval = 0.2
    exercises = ['highKnee','torsoTwist']
    randIndex = random.randint(0,1)
    prefix = exercises[randIndex]

    groupNumber = int(recording_time / recording_interval)
    # Create lists to store images and body coordinates
    body = []
    images = []

    for i in range(groupNumber):
        body.append([])
        images.append([])


    textCountdown(4, 'Do '+ prefix + ' start in :')
    start_time = time.time()
    while True:
        if ((time.time() - start_time) >= recording_time):
            break
        # ret_val, image = vs.read()
        group_index = int(np.floor((time.time() - start_time) / recording_interval))

        image = vs.read()
        images[group_index].append(image)
        # logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        temp = []

        for i in range(len(humans)):
            if i == 0:
                for j in range(common.CocoPart.Background.value):
                    if j not in humans[i].body_parts.keys():
                        temp.append(0.0)
                        temp.append(0.0)
                        continue
                    body_part = humans[i].body_parts[j]
                    coord = [body_part.x, body_part.y]
                    temp.append(coord[0])
                    temp.append(coord[1])
                body[group_index].append(temp)

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)


        # cv2.putText(image,str(len(HumanFrames)),(10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera image', image)
        if cv2.waitKey(1) == 27:
            break
    movementVerefication(groupNumber,prefix, body, images)
    del body
    del images

def getDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

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


    vs = WebcamVideoStream(cam).start()
    print("WebcamVideoStream instance 'vs' is created")

    print('Let us do some exercises')
    print('************************')

    opening = vs.read()

    cv2.imshow('Stored image', opening)
    cv2.imshow('Camera image', opening)

    textCountdown(5,' ')
    for i in range(10):
        print('round:'+ str(i))
        exerciseCaputre(vs)



    cv2.destroyAllWindows()
    vs.stop()
