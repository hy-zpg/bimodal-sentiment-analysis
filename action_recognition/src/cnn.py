#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 13:18:34 2017

@author: caetano
"""

import tensorflow as tf

class cnn:
    def __init__(self, sess, input_dim, channels):
        self.sess = sess
        self.input_dim = input_dim
        self.channels = channels
        input_shape = input_dim**2 * channels
        self.x = tf.placeholder(tf.float32, shape=[None, input_shape])

    
    # initializes weights of a given filter
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.03 )
        return tf.Variable(initial)
    
    # initializes a bias array
    def bias_variable(self, shape):
        initial = tf.constant(0.03, shape=shape)
        return tf.Variable(initial)
    
    # performs a convolution with the given filter and bias
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    # performs maxpooling
    def max_pool_2x2(self, x, pool_dim):
        return tf.nn.max_pool(x, ksize=[1, pool_dim, pool_dim, 1],
                            strides=[1, pool_dim, pool_dim, 1], padding='SAME')
    
    
    def buildCNN(self, filter_dim, pool_dim, nfilters_cnn, nfilters_fc, dropout):
        neurons_prev = self.channels
        prev_output = tf.reshape(self.x, [-1,self.input_dim,self.input_dim,self.channels])
        prev_dim = self.input_dim
        # Building convolutional layers
        for neurons in nfilters_cnn:
            W_conv = self.weight_variable([filter_dim, filter_dim,neurons_prev, neurons])
            b_conv = self.bias_variable([neurons])
            h_conv = tf.nn.relu(self.conv2d(prev_output, W_conv) + b_conv)
            h_pool = self.max_pool_2x2(h_conv, pool_dim)
            prev_output = h_pool
            neurons_prev = neurons
            prev_dim = prev_dim / pool_dim
        
        prev_dim = prev_dim*prev_dim*neurons_prev
        prev_output = tf.reshape(prev_output, [-1, prev_dim])
        # Building fully-connected layers
        for fcneurons in nfilters_fc:
            W_fc = self.weight_variable([prev_dim, fcneurons])
            b_fc = self.bias_variable([fcneurons])
            h_fc = tf.nn.relu(tf.matmul(prev_output, W_fc) + b_fc)
            #Dropout



