import os

import numpy as np
from keras.callbacks import Callback
from matplotlib import pyplot as plt


# %% Useful callbacks
class PlotResuls(Callback):
    '''This callback plots the losses and the prediction of the network during training'''
    def __init__(self, loss_each=1, saveloss=None, filepath=''):
        self.loss_each = loss_each  # Show the losses each N epochs
        self.saveloss = saveloss  # Save the losses figure
        self.filepath = filepath

    def on_train_begin(self, logs=None):
        '''Create the figure and initialise the loss arrays'''
        self.fig_loss = plt.figure('Losses')
        self.fig_acc = plt.figure('Accuracy')
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        '''Store the losses at each epoch'''
        for key, elem in logs.items():
            try:
                self.history[key].append(elem)
            except KeyError:
                self.history[key] = [elem]

        # Plot the loss
        if self.loss_each > 0 and (epoch+1) % self.loss_each == 0:
            for key, elem in logs.items():
                if 'val_' in key:
                    # Validation metrics are plotted with the training ones
                    continue

                plt.figure(key)
                plt.clf()
                plt.plot(self.history[key], label=key)
                if 'val_' + key in logs.keys():
                    plt.plot(self.history['val_' + key], label='val_' + key)

                if epoch > 100:
                    plt.gca().set_xscale('log')
                plt.legend()
                plt.pause(0.01)
                if self.saveloss:
                    plt.savefig(os.path.join(self.filepath, '%s_%s.png' % (self.saveloss, key)))
