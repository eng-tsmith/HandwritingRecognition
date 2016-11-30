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
import csv


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
    def __init__(self, test_func, inputgen, output_dir):
        self.test_func = test_func
        self.output_dir = os.path.join(output_dir)
        self.output_dir_weights = os.path.join(output_dir, "weights/")
        self.input_gen = inputgen
        self.true_string = []
        self.pred = []
        fields_title = ["True", "Pred", "CER", "CER_norm", "WER"]
        with open(os.path.join(self.output_dir, "report.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerow(fields_title)

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
        print("Reporter Callback Aufruf")

        # Save weights
        print("Saving weights to: ", os.path.join(self.output_dir_weights, 'weights%02d.h5' % epoch))
        self.model.save_weights(os.path.join(self.output_dir_weights, 'weights%02d.h5' % epoch))  #TODO

        # Get next Validation Set
        next_set = next(self.input_gen)[0]

        # Get predicted string
        decoded_res = self.decode_batch(next_set['the_input'])
        dec_string = []
        for res in decoded_res:
            dec_string.append("".join(res))
        self.pred = dec_string

        # Get true string
        sources = next_set['source_str']
        is_string = []
        for zahlen in sources:
            letters = []
            for zahl in zahlen[0]:
                letters.append(char_alpha.chars[zahl])
            is_string.append("".join(letters))
        self.true_string = is_string

        # Calc metrics  #TODO
        CER = []
        CER_norm = []
        WER = []

        #New Epoch
        with open(os.path.join(self.output_dir, "report.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerow("-")
        # Iteratre thorugh val data
        for i in range(len(self.pred)):
            edit_dist = float(editdistance.eval(self.pred[i], self.true_string[i]))
            edit_dist_norm = edit_dist / float(len(self.true_string[i]))

            CER.append(edit_dist)
            CER_norm.append(edit_dist_norm)
            WER.append(wer(self.pred[i], self.true_string[i]))
            # print('Truth: ', self.true_string[i], '   <->   Decoded: ', self.pred[i], 'CER: ', CER[i], 'CER_norm: ', CER_norm[i], 'WER: ', WER[i], 'WER_lib: ', WER_lib[i])
            print(
                "{0:<3s} {1:<20s} {2:<3s} {3:<20s} {4:<3s} {5:6.2f} {6:<3s} {7:6.2f} {8:<3s} {9:6.2f}".format('Truth:',
                                                                                                              self.true_string[i],
                                                                                                              '<-> Decoded:',
                                                                                                              self.pred[i],
                                                                                                              'CER:',
                                                                                                              CER[i],
                                                                                                              'CER_norm:',
                                                                                                              CER_norm[
                                                                                                                  i],
                                                                                                              'WER:',
                                                                                                              WER[i]))

            # ["True", "Pred", "CER", "CER_norm", "WER_lib"]
            fields_data = [self.true_string[i], self.pred[i], CER[i], CER_norm[i], WER[i]]
            with open(os.path.join(self.output_dir, "report.csv"), "a") as f:
                writer = csv.writer(f)
                writer.writerow(fields_data)
