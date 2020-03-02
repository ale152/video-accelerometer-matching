import json
import os
import numpy as np

from time import sleep
from keras import Model, Input
from keras.callbacks import Callback
from keras.layers import Lambda, Concatenate, Reshape
from keras.utils import Sequence
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


class VidAccSequence(Sequence):
    def __init__(self, validgen):
        self.validgen = validgen

    def __len__(self):
        return len(self.validgen)

    def __getitem__(self, idx):
        data, label = self.validgen.__getitem__(idx)
        vid, box, pacc, _ = data
        return [vid, box, pacc, label], label


class AuROC(Callback):
    def __init__(self, model, validgen, model_name, patience, filepath=''):
        self.validgen = validgen
        self.model_name = model_name
        self.patience = patience
        self.current_model = model
        self.filepath = filepath

        n_ft = model.layers[6].get_output_shape_at(0)[1]
        vidbox_encoder = Reshape((n_ft, 1))(model.layers[6].output)
        acc_encoder = Reshape((n_ft, 1))(model.layers[7].get_output_at(1))
        concatenated = Concatenate(axis=2)([vidbox_encoder, acc_encoder])

        inputs = model.inputs[:3]
        input_labels = Input(shape=(2, ))
        inputs.append(input_labels)
        labels = Lambda(lambda x: x)(input_labels)
        outputs = [concatenated, labels]

        self.encoder = Model(inputs=inputs, outputs=outputs)

    def _calculate_features(self):
        video_sequence = VidAccSequence(self.validgen)
        features, all_labels = self.encoder.predict_generator(video_sequence, use_multiprocessing=True, workers=4,
                                                              verbose=1)

        vidbox_ft = features[..., 0]
        acc_ft = features[..., 1]

        return vidbox_ft, acc_ft, all_labels

    def _save_model(self, appendix):
        saved = False
        while not saved:
            try:
                self.current_model.save_weights(os.path.join(self.filepath, appendix + self.model_name))
                saved = True
            except Exception as error:
                print('Error while trying to save the model {}. Trying again...'.format(error))
                sleep(5)

    def on_train_begin(self, logs=None):
        # Create the figure and initialise the loss arrays
        self.figure = plt.figure('ROC')
        self.history = {'rand': [],
                        'prev': [],
                        'dsda': [],
                        'dssa': [],
                        'sssa': []}
        self.best_target = -np.inf
        self.best_prev = -np.inf
        self.best_dsda = -np.inf
        self.best_dssa = -np.inf
        self.best_sssa = -np.inf
        self.early_stopper = 0

    def on_epoch_end(self, epoch, logs=None):
        vidbox_ft, acc_ft, all_labels = self._calculate_features()

        # ROC with random negatives
        y_vid = np.vstack((vidbox_ft, vidbox_ft))
        bf = np.random.permutation(acc_ft.shape[0])
        y_acc = np.vstack((acc_ft, acc_ft[bf, :]))
        y_score = np.sqrt(np.sum(np.square(y_vid - y_acc), axis=1))
        y_true = np.vstack((np.ones((vidbox_ft.shape[0], 1)),
                            np.zeros((acc_ft.shape[0], 1))))
        fpr, tpr, thresholds = roc_curve(y_true, -y_score)
        auc = roc_auc_score(y_true, -y_score)
        self.history['rand'].append(auc)

        # ROC with previous clip
        bf = np.argsort(self.validgen.mapping_permutation)
        y_vid = np.vstack((vidbox_ft[bf], vidbox_ft[bf]))
        y_acc = np.vstack((acc_ft[bf], acc_ft[bf][np.arange(len(acc_ft)) - 1]))
        y_score = np.sqrt(np.sum(np.square(y_vid - y_acc), axis=1))
        y_true = np.vstack((np.ones((vidbox_ft.shape[0], 1)),
                            np.zeros((acc_ft.shape[0], 1))))
        fpr, tpr, thresholds = roc_curve(y_true, -y_score)
        auc = roc_auc_score(y_true, -y_score)
        self.history['prev'].append(auc)

        # ROC with DSDA
        n_ft = acc_ft.shape[0]
        y_vid = np.vstack((vidbox_ft, vidbox_ft))
        y_acc = np.vstack((acc_ft, acc_ft))
        for i in range(n_ft):
            this_act, this_subj = all_labels[i, ]
            sel = np.logical_and(np.not_equal(all_labels[:, 0], this_act),
                           np.not_equal(all_labels[:, 1], this_subj))
            poss = np.arange(0, n_ft)
            choice = np.random.choice(poss[sel])
            y_acc[n_ft+i, ] = acc_ft[choice, ]

        y_score = np.sqrt(np.sum(np.square(y_vid - y_acc), axis=1))
        y_true = np.vstack((np.ones((vidbox_ft.shape[0], 1)),
                            np.zeros((acc_ft.shape[0], 1))))
        fpr, tpr, thresholds = roc_curve(y_true, -y_score)
        auc = roc_auc_score(y_true, -y_score)
        self.history['dsda'].append(auc)

        # ROC with DSSA
        n_ft = acc_ft.shape[0]
        y_vid = np.vstack((vidbox_ft, vidbox_ft))
        y_acc = np.vstack((acc_ft, acc_ft))
        for i in range(n_ft):
            this_act, this_subj = all_labels[i, ]
            sel = np.logical_and(np.equal(all_labels[:, 0], this_act),
                           np.not_equal(all_labels[:, 1], this_subj))
            poss = np.arange(0, n_ft)
            choice = np.random.choice(poss[sel])
            y_acc[n_ft+i, ] = acc_ft[choice, ]

        y_score = np.sqrt(np.sum(np.square(y_vid - y_acc), axis=1))
        y_true = np.vstack((np.ones((vidbox_ft.shape[0], 1)),
                            np.zeros((acc_ft.shape[0], 1))))
        fpr, tpr, thresholds = roc_curve(y_true, -y_score)
        auc = roc_auc_score(y_true, -y_score)
        self.history['dssa'].append(auc)

        # ROC with SSSA
        n_ft = acc_ft.shape[0]
        y_vid = np.vstack((vidbox_ft, vidbox_ft))
        y_acc = np.vstack((acc_ft, acc_ft))
        for i in range(n_ft):
            this_act, this_subj = all_labels[i, ]
            sel = np.logical_and(np.equal(all_labels[:, 0], this_act),
                           np.equal(all_labels[:, 1], this_subj))
            poss = np.arange(0, n_ft)
            choice = np.random.choice(poss[sel])
            y_acc[n_ft+i, ] = acc_ft[choice, ]

        y_score = np.sqrt(np.sum(np.square(y_vid - y_acc), axis=1))
        y_true = np.vstack((np.ones((vidbox_ft.shape[0], 1)),
                            np.zeros((acc_ft.shape[0], 1))))
        fpr, tpr, thresholds = roc_curve(y_true, -y_score)
        auc = roc_auc_score(y_true, -y_score)
        self.history['sssa'].append(auc)

        plt.clf()
        plt.plot(self.history['rand'], label='Random negative')
        plt.plot(self.history['prev'], label='Previous clip')
        plt.plot(self.history['dsda'], label='DSDA')
        plt.plot(self.history['dssa'], label='DSSA')
        plt.plot(self.history['sssa'], label='SSSA')
        plt.legend()

        plt.savefig(os.path.join(self.filepath, 'ROC_history.png'))

        any_improvement = False
        # Save best model based on different AUC targets
        if self.history['prev'][-1] > self.best_prev:
            print('Best AUC PREV improved from {} to {}'.format(self.best_prev, self.history['prev'][-1]))
            self.best_prev = self.history['prev'][-1]
            self._save_model('AUC_PREV_')
            any_improvement = True

        if self.history['dsda'][-1] > self.best_dsda:
            print('Best AUC DSDA improved from {} to {}'.format(self.best_dsda, self.history['dsda'][-1]))
            self.best_dsda = self.history['dsda'][-1]
            self._save_model('AUC_DSDA_')
            any_improvement = True

        if self.history['dssa'][-1] > self.best_dssa:
            print('Best AUC DSSA improved from {} to {}'.format(self.best_dssa, self.history['dssa'][-1]))
            self.best_dssa = self.history['dssa'][-1]
            self._save_model('AUC_DSSA_')
            any_improvement = True

        if self.history['sssa'][-1] > self.best_sssa:
            print('Best AUC SSSA improved from {} to {}'.format(self.best_sssa, self.history['sssa'][-1]))
            self.best_sssa = self.history['sssa'][-1]
            self._save_model('AUC_SSSA_')
            any_improvement = True

        # Save best model based on average of DSDA and DSSA
        auc_target = (self.history['dsda'][-1] + self.history['dssa'][-1]) / 2
        if auc_target > self.best_target:
            print('Best AUC improved from {} to {}'.format(self.best_target, auc_target))
            self.best_target = auc_target
            self._save_model('AUC_')
            any_improvement = True

        if any_improvement:
            self.early_stopper = 0
        else:
            self.early_stopper += 1

        if self.early_stopper == self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True

        saved = False
        while not saved:
            try:
                with open(os.path.join(self.filepath, 'auc_history.txt'), 'w') as f:
                    json.dump(self.history, f, indent=True)
                    saved = True
            except Exception as error:
                print('Error while trying to save the auc history: {}'.format(error))
                sleep(1)
