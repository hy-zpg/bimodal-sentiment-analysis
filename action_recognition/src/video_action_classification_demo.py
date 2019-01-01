import numpy as np
import keras
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Dense
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.regularizers import l2 
import pickle 
import scipy.misc
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt  
from keras.utils.vis_utils import plot_model 
from stacked_keras_model import model_generate

import cv2
from keras.models import load_model
import numpy as np

action_model_path = '../trained_models/model_generate.hdf5'