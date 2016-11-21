import numpy as np
from keras import backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Layer, Dense, Activation, Flatten
from keras.layers import Reshape, Lambda, merge, Permute, TimeDistributed
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from InputGenerator.preprocessor import prep_run
import InputGenerator.inputiterator as ii
import keras.callbacks
import InputGenerator.data_config as data
from itertools import cycle


def pad_sequence_into_array(image, maxlen):
    """

    :param image:
    :param maxlen:
    :return:
    """
    value = 0.
    image_ht = image.shape[0]

    Xout = np.ones(shape=[image_ht, maxlen], dtype=image[0].dtype) * np.asarray(value, dtype=image[0].dtype)

    trunc = image[:, :maxlen]

    Xout[:, :trunc.shape[1]] = trunc

    return Xout


def pad_label_with_blank(label, blank_id, max_length):
    """

    :param label:
    :param blank_id:
    :param max_length:
    :return:
    """
    label_len_1 = len(label[0])
    label_len_2 = len(label[0])

    label_pad = []
    # label_pad.append(blank_id)
    for _ in label[0]:
        label_pad.append(_)
        # label_pad.append(blank_id)

    while label_len_2 < max_length:
        label_pad.append(-1)
        label_len_2 += 1

    label_out = np.ones(shape=[max_length]) * np.asarray(blank_id)

    trunc = label_pad[:max_length]
    label_out[:len(trunc)] = trunc

    return label_out, label_len_1


class InputGenerator(keras.callbacks.Callback):
    def __init__(self, minibatch_size, img_w, img_h, downsample_width, val_split, output_size, absolute_max_string_len):
        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.downsample_width = downsample_width
        self.val_split = val_split
        self.output_size = output_size
        self.absolute_max_string_len = absolute_max_string_len

        self.cur_train_index = 0
        self.data_train = []
        self.data_test = []

    def load_words(self):
        # load the IAM Dataset
        self.data_train = cycle(ii.input_iter_run_train(self.minibatch_size))
        self.data_test = cycle(ii.input_iter_run_test(self.minibatch_size))
        return 0

    def get_batch(self, size, train):
        import ipdb
        ipdb.set_trace()
        batch_size = size

        #######################
        # 1. InputIterator Zeug
        if train:
            input_iterator = self.data_train.__next__()  # get from train data
        else:
            input_iterator = self.data_test.__next__()  # get from test data

        #######################
        # 2. Preprocessor Zeug
        preprocessed_input = prep_run(input_iterator, 0)

        #######################
        # 3. Predictor Zeug
        # Define input shapes
        if K.image_dim_ordering() == 'th':
            in1 = np.ones([batch_size, 1, self.img_h, self.img_w])
        else:
            in1 = np.ones([batch_size, self.img_h, self.img_w, 1])
        in2 = np.ones([batch_size, self.absolute_max_string_len])
        in3 = np.zeros([batch_size, 1])
        in4 = np.zeros([batch_size, 1])

        # Define dummy output shape
        out1 = np.zeros([batch_size])

        # Pad/Cut all input to network size
        for idx, inp in enumerate(preprocessed_input):
            x_padded = pad_sequence_into_array(inp[0], self.img_w)
            y_with_blank, y_len = pad_label_with_blank(np.asarray(inp[1]), self.output_size,
                                                       self.absolute_max_string_len)

            # Prepare input for model
            if K.image_dim_ordering() == 'th':
                in1[idx, 0, :, :] = np.asarray(x_padded, dtype='float32')[:, :]
            else:
                in1[idx, :, :, 0] = np.asarray(x_padded, dtype='float32')[:, :]
            in2[idx, :] = np.asarray(y_with_blank, dtype='float32')
            in3[idx, :] = np.array([self.downsample_width], dtype='float32')
            in4[idx, :] = np.array([y_len], dtype='float32')

        # Dictionary for Keras Model Input
        inputs = {'the_input': in1,
                  'the_labels': in2,
                  'input_length': in3,
                  'label_length': in4}
        outputs = {'ctc': out1}
        return inputs, outputs

    def on_train_begin(self, logs={}):
        # Load words
        self.load_words()

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
            # self.cur_train_index += self.minibatch_size
            # if self.cur_train_index >= self.val_split:
            #     self.cur_train_index = self.cur_train_index % self.minibatch_size
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
            # self.cur_val_index += self.minibatch_size
            # if self.cur_val_index >= self.num_words:
            #     self.cur_val_index = self.val_split + self.cur_val_index % self.minibatch_size
            yield ret
