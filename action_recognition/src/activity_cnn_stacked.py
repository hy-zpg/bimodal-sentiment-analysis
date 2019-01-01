# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 10:49:25 2017

@author: caetano
"""

import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.nan)



stackLen = 3
maxPartsTrain = 66
maxPartsTest = 31
maxFrames = 30   


# initializes weights of a given filter
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.03 )
  return tf.Variable(initial)

# initializes a bias array
def bias_variable(shape):
  initial = tf.constant(0.03, shape=shape)
  return tf.Variable(initial)

# performs a convolution with the given filter and bias
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# performs maxpooling
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def prepareData():
    train_data   = np.load( '../data/original/train_data.npy' )
    train_labels = np.load( '../data/original/train_labels.npy' )
    train_sizes  = np.load( '../data/original/train_sizes.npy' )
    test_data   = np.load( '../data/original/test_data.npy' )
    test_labels = np.load( '../data/original/test_labels.npy' )
    test_sizes  = np.load( '../data/original/test_sizes.npy' )
      
    
    train_data_clean = np.zeros((maxPartsTrain, maxFrames, train_data.shape[2]))
    train_labels_clean = np.zeros((maxPartsTrain, maxFrames, 9))
    train_sizes_clean = np.zeros((maxPartsTrain))
    
    test_data_clean = np.zeros((maxPartsTest, maxFrames, test_data.shape[2]))
    test_labels_clean = np.zeros((maxPartsTest, maxFrames, 9))
    test_sizes_clean = np.zeros((maxPartsTest))
    
    # Videos will be split into parts, including no labels
    # Train videos
    partId = 0
    frameId_clean = 0
    for videoId in range(train_data.shape[0]):
        for frameId in range(int(train_sizes[videoId])-1):
            if train_labels[videoId,frameId,9] == 0 or \
                        train_labels[videoId,frameId+1,9] == 0:
                train_data_clean[partId,frameId_clean] = train_data[videoId,frameId]
                train_labels_clean[partId,frameId_clean] = train_labels[videoId,frameId,:9]
                frameId_clean += 1
            elif frameId_clean > stackLen:
                train_sizes_clean[partId] = frameId_clean
                partId += 1
                frameId_clean = 0
        if frameId_clean > stackLen:
            train_sizes_clean[partId] = frameId_clean
    # Test videos
    partId = 0
    frameId_clean = 0
    for videoId in range(test_data.shape[0]):
        for frameId in range(int(test_sizes[videoId])-1):
            if test_labels[videoId,frameId,9] == 0 or \
                        test_labels[videoId,frameId+1,9] == 0:
                test_data_clean[partId,frameId_clean] = test_data[videoId,frameId]
                test_labels_clean[partId,frameId_clean] = test_labels[videoId,frameId,:9]
                frameId_clean += 1
            elif frameId_clean > stackLen:
                test_sizes_clean[partId] = frameId_clean
                partId += 1
                frameId_clean = 0
        if frameId_clean > stackLen:
            test_sizes_clean[partId] = frameId_clean
    return train_data_clean, train_labels_clean, train_sizes_clean,\
           test_data_clean, test_labels_clean, test_sizes_clean


def buildNetwork():
    sess = tf.InteractiveSession()
    
    x = tf.placeholder(tf.float32, shape=[None, 64*64*stackLen])
    y_ = tf.placeholder(tf.float32, shape=[None, 9])
    
    # 1st convolutional layer
    W_conv1 = weight_variable([5, 5, stackLen, 20])
    b_conv1 = bias_variable([20])
    x_image1 = tf.reshape(x, [-1,64,64,stackLen])
    h_conv1 = tf.nn.relu(conv2d(x_image1, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    # 2nd convolutional layer
    W_conv2 = weight_variable([5, 5, 20, 40])
    b_conv2 = bias_variable([40])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    # 3rd convolutional layer
    W_conv3 = weight_variable([5, 5, 40, 70])
    b_conv3 = bias_variable([70])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    
    # 4th convolutional layer
    W_conv4 = weight_variable([5, 5, 70, 100])
    b_conv4 = bias_variable([100])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)
    
    # 5th convolutional layer
    W_conv5 = weight_variable([5, 5, 100, 130])
    b_conv5 = bias_variable([130])
    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_pool5 = max_pool_2x2(h_conv5)
    
    # 1st densely connected layer
    W_fc1 = weight_variable([2 * 2 * 130, 1024])
    b_fc1 = bias_variable([1024])
    h_pool5_flat = tf.reshape(h_pool5, [-1, 2*2*130])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)
    
    # 1st dropout
    keep_prob1 = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)
    
    # 2nd densely connected layer
    W_fc2 = weight_variable([1024, 512])
    b_fc2 = bias_variable([512])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    # 2nd dropout
    keep_prob2 = tf.placeholder(tf.float32)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob2)
    
    # Output layer
    W_fc3 = weight_variable([512, 9])
    b_fc3 = bias_variable([9])
    y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
    
    return sess, x, y_, y_conv, keep_prob1, keep_prob2


def evaluate(sess, accuracy, confusion, x, y_, keep_prob1, keep_prob2,
             test_data, test_labels, test_sizes, logPath):
    accuracy_list = list()
    confusion_matrix = np.zeros((9,9))
    # for each video part
    for videoId in range(test_data.shape[0]):
        video_data = np.zeros((int(test_sizes[videoId]), stackLen, 64*64))
        video_labels = np.zeros((int(test_sizes[videoId]), 9))
        
        # for each frame of the video
        for frameId in range(int(test_sizes[videoId]) - stackLen):
            if np.sum(test_labels[videoId, frameId]) == 0:
                continue
            stack_labels = list()
            # groups stackLen frames
            for j in range(stackLen):
                video_data[frameId,j] = test_data[videoId, frameId+j]
                stack_labels += [np.argmax(test_labels[videoId, frameId+j])]
            # label referred to the last frame of the stack
            label = stack_labels[stackLen-1]
            video_labels[frameId, label] = 1
        video_data = np.reshape(video_data, (int(test_sizes[videoId]), stackLen*64*64))
        # runs networkon test batch to obtain its accuracy and confusion matrix
        [acc_batch, cm_batch] = sess.run([accuracy, confusion], feed_dict={
                x: video_data, y_: video_labels,
                keep_prob1: 1.0, keep_prob2: 1.0})
        # concatenates accuracy and sums confusion matrix
        accuracy_list += [acc_batch]
        confusion_matrix += cm_batch
    # saves test accuracy to file
    with open(logPath, 'a') as logFile:
        logFile.write(str(np.mean(accuracy_list)) + '\n')
    # presents result
    print("test accuracy %g"%(np.mean(accuracy_list)))
    print('confusion matrix:\n', confusion_matrix)


if __name__ == '__main__':    
    logPath = '../results/activity_cnn_stacked_new.txt'
    
    # splits videos into parts, removing parts related to no activity
    train_data, train_labels, train_sizes, test_data, test_labels, test_sizes = prepareData()
    # builds the neural network
    sess, x, y_, y_conv, keep_prob1, keep_prob2 = buildNetwork()
    
    # cost function
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
    # optimizes with the ADAM optimizer
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
    # prediction is correct if argmax(predicted) == argmax(ground truth)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    # nodes to calculate accuracy and confusion matrix
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    confusion = tf.confusion_matrix(labels = tf.argmax(y_,1),
                                    predictions = tf.argmax(y_conv,1),
                                    num_classes = 9)
    # initialize variables
    sess.run(tf.global_variables_initializer())
    
    train_batch_size = 32
    for epoch in range(12500):
        batch_data = np.zeros((train_batch_size, stackLen, 64*64))
        batch_labels = np.zeros((train_batch_size, 9))
        # selects video parts randomly to compose a batch for training
        videosId = np.random.choice(train_data.shape[0], train_batch_size)
        # for each video part selected
        for i in range(train_batch_size):
            # Select a frame of the video
            frameId = np.random.randint(0,train_sizes[videosId[i]]-stackLen)
            # The label must be a valid activity
            while np.sum(train_labels[frameId+stackLen-1]) == 0.0:
                frameId = np.random.randint(0,train_sizes[videosId[i]]-stackLen)
            stack_labels = list()
            # groups stackLen frames
            for j in range(stackLen):
                batch_data[i,j] = train_data[videosId[i], frameId+j]
                stack_labels += [np.argmax(train_labels[videosId[i], frameId+j])]
            # label referred to the last frame of the stack
            label = stack_labels[stackLen-1]
            batch_labels[i, label] = 1
        # prepare batch for training
        batch_data = np.reshape(batch_data, (train_batch_size, stackLen*64*64))
        # train the selected batch
        [_, cross_entropy_py] = sess.run([train_step, cross_entropy],
                                         feed_dict={x: batch_data, y_: batch_labels,
                                                    keep_prob1: 0.5, keep_prob2: 0.5})
        if epoch%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch_data, 
                                                      y_:batch_labels, 
                                                      keep_prob1: 0.5, keep_prob2: 0.5})
            print('step %d, training accuracy %g, cross entropy %g'%(epoch,
                                                                     train_accuracy,
                                                                     cross_entropy_py))
        if epoch%500 == 0:
            evaluate(sess, accuracy, confusion, x, y_, keep_prob1, keep_prob2,
                         test_data, test_labels, test_sizes, logPath)
