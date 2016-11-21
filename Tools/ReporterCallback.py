import os
import itertools
import re
import datetime
import cairocffi as cairo
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
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
import Config.char_alphabet as char_alpha


class ReporterCallback(keras.callbacks.Callback):

    def __init__(self, test_func, inputgen, num_display_words=6):
        self.test_func = test_func
        self.output_dir = os.path.join(OUTPUT_DIR, datetime.datetime.now().strftime('%A, %d. %B %Y %I.%M%p'))
        self.input_gen = inputgen
        self.num_display_words = num_display_words
        os.makedirs(self.output_dir)

    def decode_batch(self, word_batch):
        chars = char_alpha.chars
        n_classes = len(chars)

        out = self.test_func([word_batch])[0]
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            # 26 is space, 27 is CTC blank char
            outstr = []

            for il, l in enumerate(out_best):
                if (l != n_classes) and (il == 0 or l != out_best[il - 1]):
                    outstr.append(chars[l])

            ret.append(outstr)
        return ret

    def on_epoch_end(self, epoch, logs={}):
        # self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % epoch))  #TODO

        print("Reporter Callback Aufruf")
        # Save weights
        # self.model.save_weights(os.path.join(self.output_dir, 'weights.h5'))  #TODO save weightsssf
        # Predict
        # word_batch = self.model.validation_data
        import ipdb
        ipdb.set_trace()
        decoded_res = self.decode_batch(next(self.input_gen)[0])

        # parse out string
        dec_string = []
        for res in decoded_res:
            out_str = []
            for c in res:
                out_str.append(c)
            dec_string.append("".join(out_str))
        self.pred = dec_string

        # Calc metric
        edit_dist = []
        mean_ed = []
        mean_norm_ed = []
        for i in range(len(self.pred)):
            edit_dist = editdistance.eval(self.pred[i], self.true_string[i])
            mean_ed = float(edit_dist)
            mean_norm_ed = float(edit_dist) / float(len(self.true_string[i]))
        # mean_ed = float(edit_dist)
        # mean_norm_ed = float(edit_dist) / float(len(self.true_string))
            self.char_error.append(mean_ed)
            self.char_error_rate.append(mean_norm_ed)
            if mean_ed == 0.0:
                self.word_error_rate.append(0)
            else:
                self.word_error_rate.append(1)
            self.WER.append(wer("".join(self.pred[i]), self.true_string[i]))
            print('Truth: ', self.true_string[i], '   <->   Decoded: ', self.pred[i]))