import Config.data_config as data
from random import shuffle


def input_iter_run_train(n_batch_size):
    """
    Generator for training with IAM data
    :param n_batch_size: batchsize
    """
    print("====== Word Training ======")
    for fold in data.dataset_words:
        inputs = []
        for input in fold:
            # print("Train with: ", input)
            inputs.append(input)
            if len(inputs) == n_batch_size:
                input_batch = inputs
                inputs = []
                test = 0
                line = 0
                shuffle(input_batch)
                yield input_batch, test, line


def input_iter_run_test(n_batch_size):
    """
    Generator for testing with IAM data
    :param n_batch_size: batchsize
    """
    print("====== Word Testing ======")
    for fold in data.dataset_test_words:
        inputs = []
        for input in fold:
            # print("Test:", input)
            inputs.append(input)
            if len(inputs) == n_batch_size:
                input_batch = inputs
                inputs = []
                test = 1
                line = 0
                shuffle(input_batch)
                yield input_batch, test, line
    # OLD:
    # print("====== Line Training ======")
    # for epoch in range(1, n_epochs_line + 1):
    #     print("Epoche: ", epoch)
    #     for fold in data.dataset_train:
    #         inputs = []
    #         for input in fold:
    #             # print("Train with: ", input)
    #             inputs.append(input)
    #             if len(inputs) == n_batch_size:
    #                 input_batch = inputs
    #                 inputs = []
    #                 test = 0
    #                 line = 1
    #                 yield input_batch, test, line, epoch
    #
    #     for fold in data.dataset_val:
    #         inputs = []
    #         for input in fold:
    #             # print("Test:", input)
    #             inputs.append(input)
    #             if len(inputs) == n_batch_size:
    #                 input_batch = inputs
    #                 inputs = []
    #                 test = 1
    #                 line = 1
    #                 yield input_batch, test, line, epoch


