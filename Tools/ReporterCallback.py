import os
import itertools
import re
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
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
import Config.char_alphabet as char_alpha


def wer(ref, hyp, debug=False):
    """
    http://progfruits.blogspot.de/2014/02/word-error-rate-wer-and-word.html
    :param ref:
    :param hyp:
    :param debug:
    :return:
    """
    r = ref.split()
    h = hyp.split()
    # costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    DEL_PENALTY = 1
    INS_PENALTY = 1
    SUB_PENALTY = 1

    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r) + 1):
        costs[i][0] = DEL_PENALTY * i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY  # penalty is always 1
                insertionCost = costs[i][j - 1] + INS_PENALTY  # penalty is always 1
                deletionCost = costs[i - 1][j] + DEL_PENALTY  # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("OK\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("SUB\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j -= 1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i -= 1
            if debug:
                lines.append("DEL\t" + r[i] + "\t" + "****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
    if len(r) > 0.0:
        wer_result = (numSub + numDel + numIns) / (float)(len(r))
    else:
        wer_result = -1
    return wer_result


class ReporterCallback(keras.callbacks.Callback):
    def __init__(self, test_func, inputgen):
        self.test_func = test_func
        # self.output_dir = os.path.join(OUTPUT_DIR, datetime.datetime.now().strftime('%A, %d. %B %Y %I.%M%p'))
        self.input_gen = inputgen
        # os.makedirs(self.output_dir)
        self.true_string = []
        self.char_error = []
        self.char_error_rate = []
        self.word_error_rate = []
        self.WER = []
        self.pred = []

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
        # self.model.save_weights(os.path.join(self.output_dir, 'weights.h5'))  #TODO save weightssfsf
        # Predict
        # word_batch = self.model.validation_data



        next_set = next(self.input_gen)[0]
        decoded_res = self.decode_batch(next_set['the_input'])
        import ipdb
        ipdb.set_trace()
        sources = next_set['source_str']

        # parse out pred string
        dec_string = []
        for res in decoded_res:
            dec_string.append("".join(res))
        self.pred = dec_string

        # parse out true string
        is_string = []
        for zahlen in sources:
            letters = []
            for zahl in zahlen:
                letters.append(char_alpha.chars[zahl])
            is_string.append("".join(letters))
        self.true_string = is_string

        import ipdb
        ipdb.set_trace()

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
            print('Truth: ', self.true_string[i], '   <->   Decoded: ', self.pred[i])