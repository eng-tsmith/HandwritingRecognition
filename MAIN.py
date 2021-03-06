import os
import sys
import datetime
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
import Tools.InputGenerator as InputGenerator
import Tools.ReporterCallback as ReporterCallback
from keras.regularizers import l2
from keras.layers.core import K  # somehow if backend imported here it works OLD: from keras import backend as K
import tensorflow as tf

# Manual deactivation of learning mode for backend functions
K.set_learning_phase(0)

# Set logging level of TF (DEBUG, INFO, WARN, ERROR, FATAL)
# tf.logging.set_verbosity(tf.logging.ERROR)


def ctc_lambda_func(args):
    """
    Here the actual CTC loss calc occurs despite it not being an internal Keras loss function
    :param args: [y_pred, labels, input_length, label_length]
    :return: ctc_batch_cost
    """
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


##########################################################
if __name__ == '__main__':
    """
    Main function to start end-to-end pipeline preprocessing, training and classification
    """
    print("===========================")
    print("===========================")
    print("Welcome to Handwriting Recognizer")
    print("===========================")
    print("===========================")

    # Experiment name and output directory
    if len(sys.argv) == 2:
        experiment = str(sys.argv[1])
    else:
        print("No experiment name. Using date:", str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        experiment = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))  #os.path.join(output_dir, datetime.datetime.now().strftime('%A, %d. %B %Y %I.%M%p')) #e
    out_dir = os.path.join(os.getcwd(), "output/", experiment)
    out_dir_weights = os.path.join(os.getcwd(), "output/", experiment, "weights/")
    out_dir_tb = os.path.join(os.getcwd(), "output/", experiment, "TB/")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_weights, exist_ok=True)
    os.makedirs(out_dir_tb, exist_ok=True)

    # Nr Epochs
    nb_epoch = 1000
    absolute_max_string_len = 40
    rnn_size = 512

    # Optimizer
    # clipnorm seems to speeds up convergence
    clipnorm = 5
    lr = 0.001
    decay = 1e-6

    sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True, clipnorm=clipnorm)
    rms = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    optimizer = rms

    # Input Parameters
    chars = char_alpha.chars
    output_size = char_alpha.size_char

    # Input Parameters
    img_h = 64
    img_w = 256 #TODO

    # Data size
    minibatch_size = 32
    nr_trainset = 6161  #TODO see trainset
    nr_testset = 1861  #TODO see testset int(words_per_epoch * val_split)
    train_words = nr_trainset - nr_trainset%minibatch_size
    val_words = nr_testset - nr_trainset%minibatch_size

    # Network parameters
    conv_num_filters_1 = 32
    conv_num_filters_2 = 64
    conv_num_filters_3 = 128
    conv_num_filters_4 = 256
    filter_size = 3
    pool_size_w = 1
    pool_size_h = 2
    time_dense_size = 32

    if K.image_dim_ordering() == 'th':
        input_shape = (1, img_h, img_w)
    else:
        input_shape = (img_h, img_w, 1)

    # the 2 is critical here since the first couple outputs of the RNN tend to be garbage:
    downsampled_width = int(img_w / (pool_size_w * pool_size_w) - 2)

    # Init Generator
    input_gen = InputGenerator.InputGenerator(minibatch_size=minibatch_size,
                                              img_w=img_w,
                                              img_h=img_h,
                                              downsample_width=downsampled_width,
                                              output_size=output_size,
                                              absolute_max_string_len=absolute_max_string_len)

    # Activition function
    act = 'relu'

    #################################################
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

    inner = Convolution2D(conv_num_filters_4, filter_size, filter_size, border_mode='same',
                          activation=act, init='he_normal', name='conv4')(inner)
    inner = MaxPooling2D(pool_size=(pool_size_h, pool_size_w), name='max4')(inner)

    # CNN to RNN convert
    time_steps = img_w / (pool_size_w * pool_size_w * pool_size_w * pool_size_w)

    conv_to_rnn_dims = ((img_h / (pool_size_h * pool_size_h * pool_size_h * pool_size_h)) * conv_num_filters_4, time_steps)
    a = conv_to_rnn_dims[0]
    b = conv_to_rnn_dims[1]
    c = [int(a), int(b)]

    # Reshape
    inner = Reshape(target_shape=c, name='reshape')(inner)

    # Normalization
    inner = normalization.BatchNormalization(name='norm')(inner) #Works well

    # Permute
    inner = Permute(dims=(2, 1), name='permute')(inner)

    # cuts down input size going into RNN:
    inner = TimeDistributed(Dense(time_dense_size, activation=act, name='dense1'))(inner)

    # RNN
    # Two layers of bidirectional LSTMs
    # 1st bidirectional LSTM
    lstm_1 = LSTM(rnn_size, return_sequences=True, init='he_normal', name='lstm1', forget_bias_init='one',
                  W_regularizer=l2(0.01), U_regularizer=l2(0.01), b_regularizer=l2(0.01),
                  dropout_W=0.1, dropout_U=0.1)(inner)# TODO
    lstm_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='lstm1_b', forget_bias_init='one',
                   W_regularizer=l2(0.01), U_regularizer=l2(0.01), b_regularizer=l2(0.01),
                   dropout_W=0.1, dropout_U=0.1)(inner)# TODO

    # Merge SUM
    lstm1_merged = merge([lstm_1, lstm_1b], mode='sum') # TODO!!!!

    # 2nd bidirectional LSTM
    lstm_2 = LSTM(rnn_size, return_sequences=True, init='he_normal', name='lstm2', forget_bias_init='one',
                  W_regularizer=l2(0.01), U_regularizer=l2(0.01), b_regularizer=l2(0.01),
                  dropout_W=0.1, dropout_U=0.1)(lstm1_merged)# TODO
    lstm_2b = LSTM(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='lstm2_b', forget_bias_init='one',
                   W_regularizer=l2(0.01), U_regularizer=l2(0.01), b_regularizer=l2(0.01),
                   dropout_W=0.1, dropout_U=0.1)(lstm1_merged)# TODO

    # transforms RNN output to character activations:
    inner = TimeDistributed(Dense(output_size + 1, init='he_normal', name='dense2'))(merge([lstm_2, lstm_2b], mode='concat')) # mode='concat')) # TODO!!!!
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

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(optimizer=optimizer, loss={'ctc': lambda y_true, y_pred: y_pred})  #, metrics=[self.tim_metric]

    # Reporter captures output of softmax so we can decode the output during visualization
    print("Init Reporter")
    test_func = K.function([input_data], [y_pred])
    reporter = ReporterCallback.ReporterCallback(test_func, input_gen.next_val(), out_dir)

    # InitTensorBoard
    # out_dir = os.path.join(os.getcwd(), "output/TF/", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print("Saving Tensorboard to: ", out_dir_tb)
    TensorBoard = keras.callbacks.TensorBoard(log_dir=out_dir_tb, histogram_freq=0, write_graph=False)

    # Init NN done
    print("Saving graph to: ", out_dir)
    plot(model, to_file=os.path.join(out_dir, 'model.png'))
    print("Compiled Keras model successfully.")

    # TRAIN NETWORK
    model.fit_generator(generator=input_gen.next_train(), samples_per_epoch=train_words,
                        nb_epoch=nb_epoch, validation_data=input_gen.next_val(), nb_val_samples=val_words,
                        callbacks=[TensorBoard, reporter])

    print("Finished training successfully.")
    print("Saving model...")
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(out_dir, "model.json"), "w") as json_file:
        json_file.write(model_json)
    print("Saved model to: ", os.path.join(out_dir, "model.json"))

