import Config.data_config as data


def input_iter_run_train(n_batch_size):
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
                yield input_batch, test, line


def input_iter_run_test(n_batch_size):
    print("====== Word Testing ======")
    for fold in data.dataset_val_words:
        inputs = []
        for input in fold:
            # print("Test:", input)
            inputs.append(input)
            if len(inputs) == n_batch_size:
                input_batch = inputs
                inputs = []
                test = 1
                line = 0
                yield input_batch, test, line

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

def sizes(self):  # TODO
    fold_lens1 = data.dataset_words_size
    fold_lens2 = data.dataset_val_words_size
    fold_lens3 = data.dataset_train_size
    fold_lens4 = data.dataset_val_size

    n_epochs_word = data.n_epochs_word
    n_epochs_line = data.n_epochs_line

    return fold_lens1, fold_lens2, fold_lens3, fold_lens4, n_epochs_word, n_epochs_line
