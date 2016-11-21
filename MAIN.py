import os
import itertools
import datetime
import editdistance
import numpy as np
from scipy import ndimage
import pylab
from keras import backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Layer, Dense, Activation, Flatten
from keras.layers import Reshape, Lambda, merge, Permute, TimeDistributed
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
from keras.utils.visualize_util import plot
import Tools.char_alphabet as char_alpha
import Tools.InputGenerator as InputGenerator


# the actual loss calc occurs here despite it not being
# an internal Keras loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# Input Parameters
chars = char_alpha.chars
output_size = len(chars)

# # Save weights
# self.test = 0

# Input Parameters
lr = 0.001
img_h = 64
img_w = 512
nb_epoch = 50
minibatch_size = 32
words_per_epoch = 1692  #TODO
val_split = 0.2
val_words = int(words_per_epoch * val_split)
absolute_max_string_len = 100

# Network parameters
conv_num_filters = 16
filter_size = 3
pool_size_1 = 4
pool_size_2 = 2
time_dense_size = 32
rnn_size = 512
time_steps = img_w / (pool_size_1 * pool_size_2)

if K.image_dim_ordering() == 'th':
    input_shape = (1, img_h, img_w)
else:
    input_shape = (img_h, img_w, 1)

downsampled_width = int(img_w / (pool_size_1 * pool_size_2) - 2)

# # Init Generator
input_gen = InputGenerator.InputGenerator(minibatch_size=32,
                                          img_w=img_w,
                                          img_h=img_h,
                                          downsample_width=downsampled_width,
                                          val_split=words_per_epoch - val_words,
                                          output_size=output_size,
                                          absolute_max_string_len=absolute_max_string_len)

# Optimizer
# clipnorm seems to speeds up convergence
clipnorm = 5
sgd = SGD(lr=lr, decay=3e-7, momentum=0.9, nesterov=True, clipnorm=clipnorm)
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# Activition functrion
act = 'relu'

#################################################
# Network archtitecture
input_data = Input(name='the_input', shape=input_shape, dtype='float32')

# CNN encoder
inner = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                      activation=act, name='conv1')(input_data)
inner = MaxPooling2D(pool_size=(pool_size_1, pool_size_1), name='max1')(inner)
inner = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                      activation=act, name='conv2')(inner)
inner = MaxPooling2D(pool_size=(pool_size_2, pool_size_2), name='max2')(inner)

# CNN to RNN convert
conv_to_rnn_dims = ((img_h / (pool_size_1 * pool_size_2)) * conv_num_filters, time_steps)
a = conv_to_rnn_dims[0]
b = conv_to_rnn_dims[1]
c = [int(a), int(b)]

inner = Reshape(target_shape=c, name='reshape')(inner)
inner = Permute(dims=(2, 1), name='permute')(inner)

# cuts down input size going into RNN:
inner = TimeDistributed(Dense(time_dense_size, activation=act, name='dense1'))(inner)

# RNN
# Two layers of bidirecitonal GRUs
# GRU seems to work as well, if not better than LSTM:
gru_1 = GRU(rnn_size, return_sequences=True, name='gru1')(inner)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, name='gru1_b')(inner)
gru1_merged = merge([gru_1, gru_1b], mode='sum')
gru_2 = GRU(rnn_size, return_sequences=True, name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True)(gru1_merged)

# transforms RNN output to character activations:
inner = TimeDistributed(Dense(output_size + 1, name='dense2'))(merge([gru_2, gru_2b], mode='concat'))
y_pred = Activation('softmax', name='softmax')(inner)
Model(input=[input_data], output=y_pred).summary()

# LABELS
labels = Input(name='the_labels', shape=[absolute_max_string_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

# CTC layer
# Keras doesn't currently support loss funcs with extra parameters
# so CTC loss is implemented in a lambda layer
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")([y_pred, labels, input_length, label_length])

# Keras Model of NN
# Model(input=[input_data, labels, input_length, label_length], output=[loss_out]).summary()

model = Model(input=[input_data, labels, input_length, label_length], output=[loss_out])

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(optimizer=rms,loss={'ctc': lambda y_true, y_pred: y_pred})  #, metrics=[self.tim_metric]

# captures output of softmax so we can decode the output during visualization
test_func = K.function([input_data], [y_pred])
# TODO metric_recorder = MetricCallback(test_func)

# Init TensorBoard
out_dir = os.path.join(os.getcwd(), "output/TF/", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(out_dir)
print("Saving Tensorboard to: ", out_dir)
TensorBoard = keras.callbacks.TensorBoard(log_dir=out_dir, histogram_freq=1, write_graph=False)  # a

# Init NN done
plot(model, to_file=os.path.join(os.getcwd(), "output/model.png"))
print("Compiled Keras model successfully.")



# TRAIN NETWORK
model.fit_generator(generator=input_gen.next_train(), samples_per_epoch=(words_per_epoch - val_words),
                    nb_epoch=nb_epoch, validation_data=input_gen.next_val(), nb_val_samples=val_words,
                    callbacks=[TensorBoard])
