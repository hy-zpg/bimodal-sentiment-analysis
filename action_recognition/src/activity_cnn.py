# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 19:42:43 2017

@author: caetano
"""

import numpy as np
import cv2
import tensorflow as tf
np.set_printoptions(threshold=np.nan)

# initializes weights of a given filter
def weight_variable(shape):
   initial = tf.truncated_normal(shape, stddev=0.01)
   return tf.Variable(initial)

# initializes a bias array
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

# performs a convolution with the given filter and bias
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# performs maxpooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# shows a frame on the screen. Used for debugging.
def showFrame(frame):
     frameImg = np.reshape(frame, (64,64))
     frameImg = np.array(frameImg, dtype = np.uint8)
     cv2.imshow('frame', frameImg)
     cv2.waitKey(0)
     
# function to flip a frame o the image, used to enrich the sample
def flipFrame(frame):
     frameImg = np.reshape(frame, (64,64))
     frameImg = np.array(frameImg, dtype = np.uint8)
     newFrame = cv2.flip(frameImg, 1)
     newFrame = np.reshape(newFrame, (64*64))
     return newFrame


def prepareData():
    # reads data
    train_data   = np.load( '../data/original/train_data.npy' )
    train_labels = np.load( '../data/original/train_labels.npy' )
    train_sizes  = np.load( '../data/original/train_sizes.npy' )
    test_data   = np.load( '../data/original/test_data.npy' )
    test_labels = np.load( '../data/original/test_labels.npy' )
    test_sizes  = np.load( '../data/original/test_sizes.npy' )
    
    # Removes all frames not related to any activity    
    # Groups all frames in a single list, without separating them by video
    all_train_frames = list()
    all_train_labels = list()
    all_test_frames = list()
    all_test_labels = list()
    
    # Train frames
    for i in range( 42 ):
        for j in range( int(train_sizes[i]) ):
            label = train_labels[i,j]
            if label[9] != 1:
                all_train_frames += [train_data[i,j]]
                all_train_frames += [flipFrame(train_data[i,j])]
                all_train_labels += [label[0:9]] + [label[0:9]]
    # Test frames
    for i in range( 20 ):
        for j in range( int(test_sizes[i]) ):
            label = test_labels[i,j]
            if label[9] != 1:
                all_test_frames += [test_data[i,j]]
                all_test_frames += [flipFrame(test_data[i,j])]
                all_test_labels += [label[0:9]] + [label[0:9]]
        
    all_train_frames = np.array(all_train_frames)
    all_train_labels = np.array(all_train_labels)
    all_test_frames  = np.array(all_test_frames)
    all_test_labels  = np.array(all_test_labels)
    return all_train_frames, all_train_labels, all_test_frames, all_test_labels


def buildNetwork():
    sess = tf.InteractiveSession()
    
    # inputs and outputs
    x = tf.placeholder(tf.float32, shape=[None, 64*64])
    y_ = tf.placeholder(tf.float32, shape=[None, 9])
    
    # 1st convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 20])
    b_conv1 = bias_variable([20])
    x_image1 = tf.reshape(x, [-1,64,64,1])
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
    
    # 1st fully connected layer
    W_fc1 = weight_variable([2 * 2 * 130, 1024])
    b_fc1 = bias_variable([1024])
    h_pool5_flat = tf.reshape(h_pool5, [-1, 2*2*130])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)
    
    # 1st dropout
    keep_prob1 = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)
    
    # 2nd fully connected layer
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
             all_test_frames, all_test_labels, logPath):
    # evaluates in batches, to avoid "resource exhausted" error
    num_batches = 32
    # gets number size of each batch
    test_batch_len = int(all_test_frames.shape[0]/num_batches)
    # initializes list for accuracies of each batch (overall = mean)
    accuracy_list = list()
    # initializes confusion matrix
    confusion_matrix = np.zeros((9,9))
    for i in range(num_batches):
        # gets the number of frames to fill the batch
        begin_pos = i*test_batch_len
        end_pos = i*test_batch_len + test_batch_len
        # runs network on the batch, to obtain accuracy and confusion matrix
        [acc_batch, cm_batch] = sess.run([accuracy, confusion], feed_dict={
                x: all_test_frames[begin_pos:end_pos], 
                y_: all_test_labels[begin_pos:end_pos],
                keep_prob1: 1.0, keep_prob2: 1.0})
        # concatenates accuracy and sums confusion matrix
        accuracy_list += [acc_batch]
        confusion_matrix += cm_batch
    # saves test accuracy to file
    with open(logPath, 'a') as logFile:
        logFile.write(str(np.mean(accuracy_list)) + '\n')
    # shows results
    print('test accuracy: %g'%(np.mean(accuracy_list)))
    print('confusion matrix:\n', confusion_matrix)



if __name__ == '__main__':    
    logPath = '../results/activity_cnn.txt'
    
    # turns data into "array of frames"
    all_train_frames, all_train_labels, all_test_frames, all_test_labels = prepareData()
    # builds the neural network
    sess, x, y_, y_conv, keep_prob1, keep_prob2 = buildNetwork()
    
    # cost function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    # optimizes with the ADAM optimizer
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # prediction is correct if argmax(predicted) == argmax(ground truth)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    # nodes to calculate accuracy and confusion matrix
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    confusion = tf.confusion_matrix(labels = tf.argmax(y_,1),
                                    predictions = tf.argmax(y_conv,1),
                                    num_classes = 9)
    # initialize variables
    sess.run(tf.global_variables_initializer())
    
    # prepares file to register test accuracies over time
    with open(logPath, 'w') as logFile:
        logFile.write('Train accuracy\n')
    # trains for a certain number of epochs
    for epoch in range(12501):
        # selects sample at random
        sample = np.random.choice(all_train_frames.shape[0], 64)
        # prepares batch data and respective labels
        batch_data = all_train_frames[sample]
        batch_labels = all_train_labels[sample]
        # train the network with the given batch
        [_, cross_entropy_py] = sess.run([train_step, cross_entropy],
                                    feed_dict={x: batch_data, y_: batch_labels,
                                    keep_prob1: 0.3, keep_prob2: 0.3})
        # shows train accuracy and entropy, for the batch, each 100 epochs
        if epoch%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch_data, y_:batch_labels,
                keep_prob1: 1.0, keep_prob2: 1.0})
            print('step %d, training accuracy %g, cross entropy %g'%\
                  (epoch, train_accuracy, cross_entropy_py))
        # shows the overall test accuracy and confusion matrix each 500 epochs
        if epoch%500==0:
            evaluate(sess, accuracy, confusion, x, y_, keep_prob1, keep_prob2,
                     all_test_frames, all_test_labels, logPath)
