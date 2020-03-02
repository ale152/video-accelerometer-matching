import os
import argparse
import numpy as np
from random import seed
from tensorflow import set_random_seed
import itertools
from keras.utils import multi_gpu_model
from keras.callbacks import LambdaCallback
from batch_generator import BatchGenerator
from plot_results import PlotResuls
from multi_gpu_checkpoint import MultiGpuCheckpoint
from network_models import triplet_network, good_distance, bad_distance, triplet_loss, rtl_loss, triplet_acc
from keras import optimizers
from test import calculate_features, calculate_ROC, plot_features, calculate_mAP, metrics_vs_n_people
from auc_roc import AuROC
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

# Dynamically grow the memory used on the GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


def get_experiment(i):
    '''Return the training parameters for the experiments tested in the paper'''
    possible_negatives = ['50_DSDA-50_DSSA', '25_DSDA-25_DSSA-50_SSSA', '100_SSSA', '50_SSSA-50_ADJ', '100_ADJ',
                          '11_DSDA-11_DSSA-11_SSDA-33_SSSA-33_ADJ']
    possible_models = ['fully_conv', 'LSTM_with_pool', 'LSTM_no_pool']
    possible_losses = [triplet_loss, rtl_loss]
    experiments = list(itertools.product(possible_negatives, possible_losses, possible_models))

    return experiments[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', action='store', required=False, type=int, default=None)
    parser.add_argument('--mode', required=False)  # Just for pycharm
    parser.add_argument('--port', required=False)  # Just for pycharm
    args = parser.parse_args()
    if args.experiment is not None:
        experiment = args.experiment
    else:
        experiment = 0

    negative_type, loss, testing_model = get_experiment(experiment)
    exp_name = f'xfin_{negative_type}_{testing_model}_{loss.__name__}'
    save_dir = os.path.join(os.getcwd(), exp_name)
    print('IDEXP: {}'.format(exp_name))
    os.makedirs(save_dir, exist_ok=True)

    # Training settings
    n_epochs = 5000  # Maximum number of training epochs
    patience_epochs = 50  # Stop training if results don't improve
    batch_size = 16
    model_name = 'triplet_matching.h5'
    learning_rate = 1e-4
    multi_gpu = True  # Train on multiple GPUs (set up for 2)
    batch_normalization = False  # Use BN, False for the results in the paper
    data_augmentation = True
    dropout = True

    # Data parameters
    acc_type = 'ACC1.000000'  # Waist: 'ACC0.000000', Wrist: 'ACC1.000000'. Only wrist is used in the paper
    data_type = 'acc_sil_100_95_clean'
    data_type_test = 'acc_sil_100_00_clean'
    subjects_train = range(1, 8)
    subjects_test = range(8, 11)
    negative_type_test = '50_DSDA-50_DSSA'

    # Only use moving activities
    activity_filter = ['walking', 'wiping', 'vacuuming', 'sweeping', 'exercising', 'stretching', 'cleaning']

    # Pre-training
    pre_training_model = False  # Used to resume previous training

    # Set random seeds
    seed(0)  # Python
    np.random.seed(0)  # Numpy
    set_random_seed(0)  # Tensorflow

    # Data paths
    data_path = os.path.join(r'/mnt/storage/scratch/user/calorie_reid', data_type)
    data_path_test = os.path.join(r'/mnt/storage/scratch/user/calorie_reid', data_type_test)
    max_queue_size = 8
    n_workers = 2
    load_zip_memory = True
    use_multiprocess = True

    # Define Keras models for potential multiple-gpu training
    if multi_gpu:
        with tf.device('/cpu:0'):
            model = triplet_network(testing_model, use_bn=batch_normalization, use_dropout=dropout)
            if pre_training_model:
                model.load_weights(os.path.join(save_dir, pre_training_model), by_name=True)

        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=optimizers.Adam(lr=learning_rate),
                               loss=loss, metrics=[good_distance, bad_distance, triplet_acc])
    else:
        model = triplet_network(testing_model, use_bn=batch_normalization, use_dropout=dropout)
        model.compile(optimizer=optimizers.Adam(lr=learning_rate),
                      loss=loss, metrics=[good_distance, bad_distance, triplet_acc])

        if pre_training_model:
            model.load_weights(os.path.join(save_dir, pre_training_model), by_name=True)
        parallel_model = model

    # Define training callbacks
    # Checkpointer saves the best model based on the minimum validation loss
    checkpointer = MultiGpuCheckpoint(model, filepath=os.path.join(save_dir, model_name),
                                      verbose=1, save_best_only=True, monitor='val_loss', save_weights_only=True)
    # This callback takes care of plotting loss, matching distance, non-matching distance and triplet accuracy during
    # training
    plot_results = PlotResuls(loss_each=1, saveloss='plots', filepath=save_dir)

    # Define the data loaders, used for training (gen) and validation (validgen)
    gen = BatchGenerator(data_folder=data_path, batch_size=batch_size, acc_folder=acc_type, name='training',
                         negative_type=negative_type, load_zip_memory=load_zip_memory, subjects=subjects_train,
                         activity=activity_filter, acc_augmentation=False, vid_augmentation=data_augmentation, shuffle=True)
    validgen = BatchGenerator(data_folder=data_path_test, batch_size=batch_size, acc_folder=acc_type, name='validating',
                              negative_type=negative_type_test, load_zip_memory=load_zip_memory, subjects=subjects_test,
                              shuffle=True, activity=activity_filter, acc_augmentation=False, vid_augmentation=False)
    # Define additional callbacks that make use of the data generators
    # This callback updates the combination of triplets (negative samples) at the end of each epoch
    update_mapping = LambdaCallback(on_epoch_end=gen.generate_mapping)
    # This callback calculates and plot the progress of the area under the ROC curves
    auc_callback = AuROC(model, validgen, model_name, patience=patience_epochs, filepath=save_dir)

    validation_steps = len(validgen)
    parallel_model.fit_generator(gen, max_queue_size=max_queue_size, validation_data=validgen,
                                 validation_steps=validation_steps, epochs=n_epochs,
                                 use_multiprocessing=use_multiprocess, workers=n_workers,
                                 callbacks=[checkpointer, plot_results, update_mapping, auc_callback])
    # Save the final model (is not the optimal model but it's useful for debugging)
    parallel_model.save(os.path.join(save_dir, 'final_' + model_name))

    # Load the best model based on the auROC values
    model.load_weights(os.path.join(save_dir, 'AUC_' + model_name))
    # Calculate the video and accelerometer features for the validation data and evaluate the results
    vidbox_ft, acc_ft, all_labels, validgen = calculate_features(model, data_path_test, batch_size, load_zip_memory,
                                                                 acc_type, activity_filter, subjects_test)
    calculate_ROC(vidbox_ft, acc_ft, all_labels, validgen, filepath=save_dir)
    plot_features(vidbox_ft, acc_ft, all_labels, validgen, filepath=save_dir)
    calculate_mAP(vidbox_ft, acc_ft, all_labels, validgen, filepath=save_dir)
    metrics_vs_n_people(vidbox_ft, acc_ft, all_labels, filepath=save_dir)
    # Save the results
    np.savez_compressed(os.path.join(save_dir, 'features'), vidbox_ft=vidbox_ft, acc_ft=acc_ft, all_labels=all_labels)

if __name__ == '__main__':
    main()