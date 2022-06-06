from __future__ import division
import os
import sys
from math import *
import numpy as np
import scipy as sp
import h5py
import time

from dataclasses import dataclass
# from sklearn import KDTree


######### Neural-Network Relevant Libraries #########

import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras import regularizers
from keras.models import load_model
import numpy as np
import statistics as stat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from numpy import linalg as LA
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from keras import initializers
from keras.layers import LeakyReLU
from keras.models import model_from_json
from keras import metrics
import keras.backend as kb

print_NNweights = 1