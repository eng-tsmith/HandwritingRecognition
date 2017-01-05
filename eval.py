import os
import sys
import datetime
from keras.models import model_from_json
import numpy as np
import Tools.preprocessor_eval as preprocessor
from keras.layers.core import K  # somehow if backend imported here it works OLD: from keras import backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Layer, Dense, Activation, Flatten, Dropout
from keras.layers import Reshape, Lambda, merge, Permute, TimeDistributed, normalization
from keras.models import Model
from keras.layers.recurrent import GRU, LSTM
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Nadam
import keras.callbacks
from keras.utils.visualize_util import plot
import Config.char_alphabet as char_alpha
# import Tools.ReporterCallback as ReporterCallback
from keras.regularizers import l2
import itertools


# Manual deactivation of learning mode for backend functions
K.set_learning_phase(0)


# the actual loss calc occurs here despite it not being
# an internal Keras loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


if __name__ == '__main__':
    # load json and create model
    experiment = "dropwork"
    file_path_model = os.path.join(os.getcwd(), "output/", experiment)
    file_path_weigths = os.path.join(file_path_model, "weights/")

    # json_file = open(os.path.join(file_path_model, "model.json"), 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()

    # Input Parameters
    chars = char_alpha.chars
    output_size = char_alpha.size_char

    # Network parameters
    conv_num_filters_1 = 32
    conv_num_filters_2 = 64
    conv_num_filters_3 = 128
    filter_size = 3
    pool_size_w = 1
    pool_size_h = 2
    time_dense_size = 32

    # Input Parameters
    img_h = 64
    img_w = 256  # TODO

    if K.image_dim_ordering() == 'th':
        input_shape = (1, img_h, img_w)
    else:
        input_shape = (img_h, img_w, 1)

    # Optimizer
    # clipnorm seems to speeds up convergence
    clipnorm = 5
    lr = 0.005
    decay = 1e-6

    sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True, clipnorm=clipnorm)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    optimizer = rms

    # Nr Epochs
    absolute_max_string_len = 40
    rnn_size = 512

    # Activition function
    act = 'relu'

    # Network archtitecture
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    # CNN encoder
    inner = Convolution2D(conv_num_filters_1, filter_size, filter_size, border_mode='same',
                          activation=act, init='he_normal', name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size_h, pool_size_w), name='max1')(inner)

    inner = Convolution2D(conv_num_filters_2, filter_size, filter_size, border_mode='same',
                          activation=act, init='he_normal', name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size_h, pool_size_w), name='max2')(inner)

    inner = Convolution2D(conv_num_filters_3, filter_size, filter_size, border_mode='same',
                          activation=act, init='he_normal', name='conv3')(inner)
    inner = MaxPooling2D(pool_size=(pool_size_h, pool_size_w), name='max3')(inner)

    # Normalization
    inner = normalization.BatchNormalization(name='norm')(inner)  # Works well

    # CNN to RNN convert
    time_steps = img_w / (pool_size_w * pool_size_w * pool_size_w)

    conv_to_rnn_dims = ((img_h / (pool_size_h * pool_size_h * pool_size_h)) * conv_num_filters_3, time_steps)
    a = conv_to_rnn_dims[0]
    b = conv_to_rnn_dims[1]
    c = [int(a), int(b)]

    inner = Reshape(target_shape=c, name='reshape')(inner)
    inner = Permute(dims=(2, 1), name='permute')(inner)

    # cuts down input size going into RNN:
    inner = TimeDistributed(Dense(time_dense_size, activation=act, name='dense1'))(inner)

    # RNN
    # Two layers of bidirectional LSTMs
    # 1st bidirectional LSTM
    lstm_1 = LSTM(rnn_size, return_sequences=True, init='he_normal', name='lstm1', forget_bias_init='one',
                  W_regularizer=l2(0.01), U_regularizer=l2(0.01), b_regularizer=l2(0.01),
                  dropout_W=0.2, dropout_U=0.2)(inner)  # TODO
    lstm_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='lstm1_b',
                   forget_bias_init='one',
                   W_regularizer=l2(0.01), U_regularizer=l2(0.01), b_regularizer=l2(0.01),
                   dropout_W=0.2, dropout_U=0.2)(inner)  # TODO

    # Merge SUM
    lstm1_merged = merge([lstm_1, lstm_1b], mode='concat')  # TODO!!!!

    # 2nd bidirectional LSTM
    lstm_2 = LSTM(rnn_size, return_sequences=True, init='he_normal', name='lstm2', forget_bias_init='one',
                  W_regularizer=l2(0.01), U_regularizer=l2(0.01), b_regularizer=l2(0.01),
                  dropout_W=0.2, dropout_U=0.2)(lstm1_merged)  # TODO
    lstm_2b = LSTM(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='lstm2_b',
                   forget_bias_init='one',
                   W_regularizer=l2(0.01), U_regularizer=l2(0.01), b_regularizer=l2(0.01),
                   dropout_W=0.2, dropout_U=0.2)(lstm1_merged)  # TODO

    # transforms RNN output to character activations:
    inner = TimeDistributed(Dense(output_size + 1, init='he_normal', name='dense2'))(
        merge([lstm_2, lstm_2b], mode='concat'))  # mode='concat')) # TODO!!!!
    # CTC Softmax
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

    # load weights into new model
    print("Loading weights...")
    model.load_weights(os.path.join(file_path_weigths, "weights275.h5")) #TODO
    print("Loaded weights to model")

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(optimizer=optimizer, loss={'ctc': lambda y_true, y_pred: y_pred})  # , metrics=[self.tim_metric]

    # Reporter captures output of softmax so we can decode the output during visualization
    print("Init Reporter")
    test_func = K.function([input_data], [y_pred])

    # print("Loading model...")
    # loaded_model = model_from_json(loaded_model_json)
    # print("Loaded model from disk")

    plot(model, to_file=os.path.join(file_path_model, 'model_eval.png'))

    input_tuple = [[('../media/nas/01_Datasets/IAM/words/c06/c06-005/c06-005-05-06.png')]]
    X = preprocessor.prep_run(input_tuple)

    if K.image_dim_ordering() == 'th':
        in1 = np.ones([1, 1, img_h, img_w])
    else:
        in1 = np.ones([1, img_h, img_w, 1])

    if K.image_dim_ordering() == 'th':
        in1[0, 0, :, :] = np.asarray(X, dtype='float32')[:, :]
    else:
        in1[0, :, :, 0] = np.asarray(X, dtype='float32')[:, :]

    out = test_func([in1])[0]

    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        # 26 is space, 27 is CTC blank char
        outstr = []

        for il, l in enumerate(out_best):
            if (l != output_size) and (il == 0 or l != out_best[il - 1]):
                outstr.append(chars[l])

        ret.append(outstr)

    dec_string = []
    for res in ret:
        dec_string.append("".join(res))

    print(dec_string)

    # # Get predicted string
    # decoded_res = decode_batch(X['the_input'])
    # dec_string = []
    # for res in decoded_res:
    #     dec_string.append("".join(res))
    # self.pred = dec_string




    # score = loaded_model.evaluate(X, Y, verbose=0)  #TODO
    # print("Score: ", score)