import numpy as np
from keras import backend as K
from Tools.preprocessor import prep_run
import Tools.inputiterator as ii
import keras.callbacks
from itertools import cycle


class InputGenerator(keras.callbacks.Callback):
    def __init__(self, minibatch_size, img_w, img_h, downsample_width, output_size, absolute_max_string_len):
        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.downsample_width = downsample_width
        self.output_size = output_size
        self.absolute_max_string_len = absolute_max_string_len

        self.cur_train_index = 0
        self.data_train = []
        self.data_test = []

        # load the IAM Dataset
        self.data_train = cycle(ii.input_iter_run_train(self.minibatch_size))
        self.data_test = cycle(ii.input_iter_run_test(self.minibatch_size))

    def get_batch(self, size, train):
        """
        Get and prepare batch for neural net.
        :param size: Batchsize
        :param train: Boolean: 1-> traing; 0->Testing
        :return: Prepared input and output for Keras model
        """
        batch_size = size
        #######################
        # 1. InputIterator Zeug
        if train:
            input_iterator = self.data_train.__next__()[0]  # get from train data
        else:
            input_iterator = self.data_test.__next__()[0]  # get from test data

        #######################
        # 2. Preprocessor
        preprocessed_input = prep_run(input_iterator, 0,  self.absolute_max_string_len, self.img_w)
        # Output = [img_noise, label_blank, label_len, label_raw]

        #######################
        # 3. Predictor Zeug
        # Define input shapes
        # 1 Image
        # 2 Label with blanks
        # 3 Input length
        # 4 Label Length
        # 5 True label

        if K.image_dim_ordering() == 'th':
            in1 = np.ones([batch_size, 1, self.img_h, self.img_w])
        else:
            in1 = np.ones([batch_size, self.img_h, self.img_w, 1])
        in2 = np.ones([batch_size, self.absolute_max_string_len])
        in3 = np.zeros([batch_size, 1])
        in4 = np.zeros([batch_size, 1])
        in5 = []

        # Define dummy output shape
        out1 = np.zeros([batch_size])

        # Pad/Cut all input to network size
        for idx, inp in enumerate(preprocessed_input):
            x_padded = inp[0]
            y_with_blank = inp[1]
            y_len = inp[2]

            # Prepare input for model
            if K.image_dim_ordering() == 'th':
                in1[idx, 0, :, :] = np.asarray(x_padded, dtype='float32')[:, :]
            else:
                in1[idx, :, :, 0] = np.asarray(x_padded, dtype='float32')[:, :]
            in2[idx, :] = np.asarray(y_with_blank, dtype='float32')
            in3[idx, :] = np.array([self.downsample_width], dtype='float32')
            in4[idx, :] = np.array([y_len], dtype='float32')
            in5.append(inp[3])

        # Dictionary for Keras Model Input
        inputs = {'the_input': in1,
                  'the_labels': in2,
                  'input_length': in3,
                  'label_length': in4,
                  'source_str': in5  # used for report only
                  }
        outputs = {'ctc': out1}
        return inputs, outputs

    def next_train(self):
        while 1:
            ret = self.get_batch(self.minibatch_size, train=True)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.minibatch_size, train=False)
            yield ret
