#!/usr/bin/env python3.5

import io
import numpy as np
from tensorflow.python.framework import ops


import pickle
import math
import cv2
import tensorflow as tf
import os
#supress low level warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


exercise_name = 'torsoTwist'
groupSize = 20 #HighKnee have 20 groups of images---it means we'll create 20 models

#load data sets
pkl_file = open(exercise_name +'Data.pkl', 'rb')
mData = pickle.load(pkl_file)
pkl_file.close()
#print(highKneeData)

pkl_file = open(exercise_name +'Label.pkl', 'rb')
mLabel = pickle.load(pkl_file)
pkl_file.close()
#print(highKneeLabel)

pkl_file = open(exercise_name +'Data2.pkl', 'rb')
mData2 = pickle.load(pkl_file)
pkl_file.close()
#print(highKneeData)

pkl_file = open(exercise_name +'Label2.pkl', 'rb')
mLabel2 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(exercise_name +'Data3.pkl', 'rb')
mData3 = pickle.load(pkl_file)
pkl_file.close()
#print(highKneeData)

pkl_file = open(exercise_name +'Label3.pkl', 'rb')
mLabel3 = pickle.load(pkl_file)
pkl_file.close()


def getDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

label =[]
data =[]
dataDist = []
for i in range(groupSize):
    label.append([])
    data.append([])
    dataDist.append([])

for i in range(groupSize):
    data[i] = (mData[i] + mData2[i] + mData3[i])
    label[i] = (mLabel[i] + mLabel2[i] + mLabel3[i])
'''
for i in range(groupSize):
    for body in data[i]:
        temp = []
        distance = getDistance(body[2],body[3],body[0],body[1])
        temp.append(distance)
        for j in range(2,18):
            distance = getDistance(body[2],body[3],body[j*2],body[j*2+1])
            print('point 2 to point '+str(j) + ' distance ='+str(distance))
            temp.append(distance)
        dataDist[i].append(temp)
print(np.shape(dataDist))
print(dataDist[0][0])       
'''


def trainModel(group_number ):
    tf.reset_default_graph()

    bodies = np.array(data[group_number])
    labels = np.array(label[group_number])

    batch_size = 250
    Label_size = 2

    y_vals0 = np.array([1 if y == 0 else 0 for y in labels])
    y_vals1 = np.array([1 if y == 1 else 0 for y in labels])

    # print(y_vals1)
    y_vals = np.array([y_vals0, y_vals1])
    #print(np.shape(y_vals))
    # print(y_vals)



    W = tf.Variable(tf.zeros([36 * 1, Label_size]))
    b = tf.Variable(tf.zeros([Label_size]))


    x = tf.placeholder(tf.float32, [None, 36 * 1])
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # intermediate y value
    # y_inter= tf.placeholder(tf.float32,[5,batch_size])

    y_ = tf.placeholder(tf.float32, [None, Label_size])
    # y_ = reshape_matmul(y_inter,batch_size)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ####################################################
    ####################################################
    #calculate the max value of y
    y_pred = tf.argmax(y, 1)

    #create saver
    saver = tf.train.Saver()

    #add saver to collection
    tf.add_to_collection('input', x)
    tf.add_to_collection('output', y_pred)

    ####################################################
    ####################################################


    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(coord=coord)

        for count in range(20000):
            # ramdomly read data from bodies and labels
            rand_index = np.random.choice(len(bodies), size=batch_size)
            # print('rand_index generated')
            rand_x = bodies[rand_index]
            # print('rand_x generated')
            # print(np.shape(rand_x))
            rand_y = y_vals[:, rand_index]
            rand_y = np.transpose(rand_y)
            #print(rand_y)
            # read data from tfrecord
            # pose_data, label_data = sess.run([pose_batch_train, label_batch_train])

            sess.run(train_step, feed_dict={x: rand_x, y_: rand_y})

            if count % 30 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: rand_x, y_: rand_y})
                print('Iter %d, accuracy %4.2f%%' % (count, train_accuracy * 100))


        coord.request_stop()
        coord.join(threads)

        ####################################################

        save_path = './models/movement/' + exercise_name + str(group_number)


        spath = saver.save(sess, save_path)
        print("Model saved in file: %s" % spath)
        ####################################################
    sess.close()


for i in range(groupSize):
    trainModel(i)