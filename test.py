import hashlib
import json
import os
import cv2
import pandas as pd
import numpy as np
from keras import Model, Input
from keras.layers import Concatenate, Reshape, Lambda
from keras.utils import Sequence
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean, correlation
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
from batch_generator import BatchGenerator
from matplotlib import pyplot as plt, gridspec
from network_models import triplet_network


def calculate_features(model, data_path, batch_size, load_zip_memory, acc_type, activities, subjects=None, cache=False,
                       md5='', cache_mod=''):
    '''Function to calculate the features for videos and accelerometer using the deep model'''
    validgen = BatchGenerator(data_folder=data_path, batch_size=batch_size, acc_folder=acc_type, name='validating',
                             negative_type='dsda', load_zip_memory=load_zip_memory, subjects=subjects,
                             activity=activities, triplet_mode=True, shuffle=False)

    # Use cached values if the script is run more than once
    if cache and os.path.exists(os.path.join(cache, '{}_{}_{}.npz'.format(cache_mod, hashlib.md5(''.join(activities).encode('utf8')).hexdigest(), md5))):
        with np.load(os.path.join(cache, '{}_{}_{}.npz'.format(cache_mod, hashlib.md5(''.join(activities).encode('utf8')).hexdigest(), md5))) as data:
            return data['vidbox_ft'], data['acc_ft'], data['all_labels'], validgen

    n_ft = model.layers[6].get_output_shape_at(0)[1]
    vidbox_encoder = Reshape((n_ft, 1))(model.layers[6].output)
    acc_encoder = Reshape((n_ft, 1))(model.layers[7].get_output_at(1))
    concatenated = Concatenate(axis=2)([vidbox_encoder, acc_encoder])

    inputs = model.inputs[:3]
    input_labels = Input(shape=(2, ))
    inputs.append(input_labels)
    labels = Lambda(lambda x: x)(input_labels)
    outputs = [concatenated, labels]

    encoder = Model(inputs=inputs, outputs=outputs)

    class VidAccSequence(Sequence):
        def __init__(self, validgen):
            self.validgen = validgen
        def __len__(self):
            return len(self.validgen)
        def __getitem__(self, idx):
            data, label = validgen.__getitem__(idx)
            vid, box, pacc, _ = data
            return [vid, box, pacc, label], label

    video_sequence = VidAccSequence(validgen)
    features, all_labels = encoder.predict_generator(video_sequence, verbose=1)
    vidbox_ft = features[..., 0]
    acc_ft = features[..., 1]

    if cache:
        np.savez(os.path.join(cache, '{}_{}_{}.npz'.format(cache_mod, hashlib.md5(''.join(activities).encode('utf8')).hexdigest(), md5)),
                 vidbox_ft=vidbox_ft, acc_ft=acc_ft, all_labels=all_labels)

    return vidbox_ft, acc_ft, all_labels, validgen


def shigeta_features(data_path, load_zip_memory, acc_type, activities, subjects=None, cache=False):
    '''Function to calculate features for Shigeta et al.'''
    if cache and os.path.exists(os.path.join(cache, 'shigeta_{}.npz'.format(hashlib.md5(''.join(activities).encode('utf8')).hexdigest()))):
        with np.load(os.path.join(cache, 'shigeta_{}.npz'.format(hashlib.md5(''.join(activities).encode('utf8')).hexdigest()))) as data:
            return data['vidbox_ft'], data['acc_ft'], data['all_labels']

    validgen = BatchGenerator(data_folder=data_path, batch_size=1, acc_folder=acc_type, name='validating',
                             negative_type='dsda', load_zip_memory=load_zip_memory, subjects=subjects,
                             activity=activities, triplet_mode=True, shuffle=False)

    ft_size = 100
    vidbox_ft = np.zeros((0, ft_size))
    acc_ft = np.zeros((0, ft_size))
    all_labels = np.zeros((0, 2))

    for bi in tqdm(range(len(validgen))):
        data, label = validgen.__getitem__(bi)
        vid, box, pacc, nacc = data

        box_flt = savgol_filter(box[0, ...], 15, 3, deriv=2, axis=0, mode='nearest')
        box_flt = np.sum(np.square(box_flt), axis=1)
        box_flt -= box_flt.mean()
        box_flt /= (box_flt.std() + 1e-16)

        pacc = np.sum(np.square(pacc[0, ...]), axis=1)
        pacc_flt = savgol_filter(pacc, 15, 3, axis=0, mode='nearest')
        pacc_flt -= pacc_flt.mean()
        pacc_flt /= (pacc_flt.std() + 1e-16)

        vidbox_ft = np.vstack((vidbox_ft, box_flt))
        acc_ft = np.vstack((acc_ft, pacc_flt))
        all_labels = np.vstack((all_labels, label))

    if cache:
        np.savez(os.path.join(cache, 'shigeta_{}.npz'.format(hashlib.md5(''.join(activities).encode('utf8')).hexdigest())),
                 vidbox_ft=vidbox_ft, acc_ft=acc_ft, all_labels=all_labels)

    return vidbox_ft, acc_ft, all_labels


def cabrera_features(data_path, load_zip_memory, acc_type, activities, subjects=None, cache=False):
    '''Function to calculate the features for Cabrera et al.'''
    if cache and os.path.exists(os.path.join(cache, 'cabrera_{}.npz'.format(hashlib.md5(''.join(activities).encode('utf8')).hexdigest()))):
        with np.load(os.path.join(cache, 'cabrera_{}.npz'.format(hashlib.md5(''.join(activities).encode('utf8')).hexdigest()))) as data:
            return data['vidbox_ft_flt'], data['acc_ft_flt'], data['all_labels']

    validgen = BatchGenerator(data_folder=data_path, batch_size=1, acc_folder=acc_type, name='validating',
                             negative_type='dsda', load_zip_memory=load_zip_memory, subjects=subjects,
                             activity=activities, triplet_mode=True, shuffle=False)

    ft_size = 100
    vidbox_ft = np.zeros((0, ft_size))
    acc_ft = np.zeros((0, ft_size, 3))
    all_labels = np.zeros((0, 2))

    for bi in tqdm(range(len(validgen))):
        data, label = validgen.__getitem__(bi)
        vid, box, pacc, nacc = data

        flow_ft = np.zeros((ft_size))
        for fr in range(1, vid.shape[-1]):
            flow = cv2.calcOpticalFlowFarneback(vid[0, ..., fr-1]*255,
                                                vid[0, ..., fr]*255, None, 0.5, 4, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(np.square(flow[..., 0]) + np.square(flow[..., 1]))
            flow_ft[fr] = mag.mean()

        flow_ft[0] = flow_ft[1]
        flow_ft = flow_ft[1:] - flow_ft[0:-1]
        flow_ft = np.append(flow_ft, flow_ft[-1])


        vidbox_ft = np.vstack((vidbox_ft, flow_ft))
        acc_ft = np.concatenate((acc_ft, pacc[None, 0, ...]), axis=0)
        all_labels = np.vstack((all_labels, label))

    # Normalise accelerations
    acc_ft -= acc_ft.mean(axis=0)
    acc_ft /= (acc_ft.var(axis=0) + 1e-16)
    acc_ft = np.sqrt(np.square(acc_ft[:, :, 0]) + np.square(acc_ft[:, :, 1]) + np.square(acc_ft[:, :, 2]))

    # Normalise to one
    acc_ft /= acc_ft.max()
    vidbox_ft /= vidbox_ft.max()

    # Variance over sliding window
    df_acc = pd.DataFrame(acc_ft)
    acc_ft_flt = df_acc.rolling(10, axis=1, min_periods=0).var().values
    acc_ft_flt[:, 0] = acc_ft_flt[:, 1]
    df_vid = pd.DataFrame(vidbox_ft)
    vidbox_ft_flt = df_vid.rolling(10, axis=1, min_periods=0).var().values
    vidbox_ft_flt[:, 0] = vidbox_ft_flt[:, 1]

    if cache:
        np.savez(os.path.join(cache, 'cabrera_{}.npz'.format(hashlib.md5(''.join(activities).encode('utf8')).hexdigest())),
                 vidbox_ft_flt=vidbox_ft_flt, acc_ft_flt=acc_ft_flt, all_labels=all_labels)

    return vidbox_ft_flt, acc_ft_flt, all_labels


def calculate_mAP(vidbox_ft, acc_ft, all_labels, gen, metric='euclidean', filepath=''):
    '''Calculate the mAP using all the features'''
    # Calculate the distance matrix
    n = vidbox_ft.shape[0]
    dist = cdist(vidbox_ft, acc_ft, metric=metric)
    if metric == 'correlation':
        dist = -dist
    # Elements on the diagonal are the correct matches and should be the ones with the lowest distance
    # Check where the diagonal elements are instead
    order = np.argsort(dist, axis=1)
    diag = np.arange(0, n)[:, None]
    rank = np.argmax(np.equal(order, diag), axis=1)
    # There's only ONE correct sequence that can be retrieved
    m_ap = np.mean(1 / (rank + 1))
    print('mAP across all activities: {}'.format(m_ap))

    plt.figure()
    plt.imshow(dist, cmap='jet')
    plt.xlabel('Accelerometer features')
    plt.ylabel('Video features')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Distance')
    plt.title('Distance matrix between features')
    plt.savefig(os.path.join(filepath, 'distance_matrix.png'))

    # Rank per activity
    n_act = len(gen.activities)
    m_ap = np.zeros((n_act))
    m_rank = np.zeros((n_act))
    n_act = len(gen.activities)
    for i in tqdm(range(n_act)):
        sel_i = all_labels[:, 0] == i
        m_ap[i] = np.mean(1 / (1 + rank[sel_i])) * 100
        m_rank[i] = np.mean(rank[sel_i])

    plt.figure()
    plt.bar(np.arange(0, n_act), m_ap)
    plt.xticks(np.arange(0, n_act), gen.activities, rotation=45)
    plt.title('mAP by activity')
    plt.tight_layout()
    plt.savefig(os.path.join(filepath, 'map_by_activity.png'))

    plt.figure()
    plt.bar(np.arange(0, n_act), m_rank)
    plt.xticks(np.arange(0, n_act), gen.activities, rotation=45)
    plt.title('Average rank by activity')
    plt.tight_layout()
    plt.savefig(os.path.join(filepath, 'average_rank.png'))

    plt.figure()
    plt.hist(rank, 100)
    plt.savefig(os.path.join(filepath, 'histogram_rank.png'))


def calculate_threshold(vidbox_ft, acc_ft, all_labels, metric=euclidean):
    # ROC with random negatives
    n_ft = acc_ft.shape[0]
    y_vid = np.vstack((vidbox_ft, vidbox_ft, vidbox_ft))
    y_acc = np.vstack((acc_ft, acc_ft, acc_ft))
    for i in range(n_ft):
        # DSDA
        this_act, this_subj = all_labels[i, ]
        sel = np.logical_and(np.not_equal(all_labels[:, 0], this_act),
                       np.not_equal(all_labels[:, 1], this_subj))
        poss = np.arange(0, n_ft)
        choice = np.random.choice(poss[sel])
        y_acc[n_ft+i, ] = acc_ft[choice, ]

        # DSSA
        sel = np.logical_and(np.equal(all_labels[:, 0], this_act),
                       np.not_equal(all_labels[:, 1], this_subj))
        poss = np.arange(0, n_ft)
        choice = np.random.choice(poss[sel])
        y_acc[n_ft*2+i, ] = acc_ft[choice, ]

    y_score = np.sum(np.square(y_vid - y_acc), axis=1)
    y_true = np.vstack((np.ones((vidbox_ft.shape[0], 1)),
                        np.zeros((acc_ft.shape[0], 1)),
                        np.zeros((acc_ft.shape[0], 1))))

    # Find best threshold on the training data (EER)
    fpr, tpr, thresholds = roc_curve(y_true, -y_score)
    thr = -thresholds[np.argmin(np.abs(fpr - (1 - tpr)))]
    return thr


def variable_clip_length(vidbox_ft, acc_ft, all_labels, metric='euclidean', filepath='', method_names=[]):
    # Select the features from the three validation subjects (use all_labels)
    if not isinstance(vidbox_ft, list):
        vidbox_ft = [vidbox_ft]
        acc_ft = [acc_ft]

    n_methods = len(vidbox_ft)

    plt.figure()
    for mi in range(n_methods):
        val_sub = [8, 9, 10]
        val_vid = [vidbox_ft[mi][all_labels[:, 1] == bf] for bf in val_sub]
        val_acc = [acc_ft[mi][all_labels[:, 1] == bf] for bf in val_sub]

        max_n_clips = 150
        n_tests = 10000
        tot_size = [bar.shape[0] for bar in val_vid]
        mean_accuracies = []
        var_accuracies = []
        tested_clips = range(1, max_n_clips+1, 10)
        for n_clips in tqdm(tested_clips):
            # Calculate the distance matrices for each clip and then average them
            acc_test = []
            for test in tqdm(range(n_tests)):
                random_start = [np.random.randint(bf) for bf in tot_size]
                random_start = [np.minimum(random_start[bf], tot_size[bf]-n_clips) for bf in range(len(random_start))]
                dist = []
                for ci in range(n_clips):
                    sub_vid = [val_vid[s][random_start[s]+ci, ] for s in range(len(val_sub))]
                    sub_acc = [val_acc[s][random_start[s]+ci, ] for s in range(len(val_sub))]
                    dist.append(cdist(sub_vid, sub_acc))

                dist = np.dstack(dist).mean(axis=-1)
                ii, jj = linear_sum_assignment(dist)
                acc = np.mean(ii == jj)
                # order = np.argsort(dist, axis=1)[:, 0]
                # acc = np.mean(np.equal(order, np.arange(len(val_sub))[None, ]))
                acc_test.append(acc)

            mean_accuracies.append(np.mean(acc_test))
            var_accuracies.append(np.var(acc_test))

        name = method_names[mi] if method_names else ''
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        plt.plot(np.array(tested_clips)*3/60, mean_accuracies, label=name, color=colors[mi])
        plt.fill_between(np.array(tested_clips)*3/60, np.array(mean_accuracies)-np.array(var_accuracies)/2, np.array(mean_accuracies)+np.array(var_accuracies)/2,
                         color=colors[mi], alpha=0.3)
        plt.pause(0.1)

    plt.legend()
    plt.xlabel('Observation time (minutes)')
    plt.ylabel('Assignment accuracy')
    plt.savefig(os.path.join(filepath, 'variable_clip_length.pdf'), bbox_inches='tight', pad_inches=0)


def metrics_vs_n_people(vidbox_ft, acc_ft, all_labels, metric='euclidean', filepath='', method_names=[]):
    # Calculate the distance matrix
    if not isinstance(vidbox_ft, list):
        vidbox_ft = [vidbox_ft]
        acc_ft = [acc_ft]

    n_methods = len(vidbox_ft)

    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    for mi in range(n_methods):
        n_tests = 5
        n_ft = vidbox_ft[mi].shape[0]
        dist = cdist(vidbox_ft[mi], acc_ft[mi], metric=metric)

        max_n_clips = 10
        mAP_dsda_one_vid = np.zeros((n_tests, max_n_clips - 1))  # - 1 because we don't test the case of 1 subject
        mAP_dssa_one_vid = np.zeros((n_tests, max_n_clips - 1))
        mAP_sssa_one_vid = np.zeros((n_tests, max_n_clips - 1))
        mAP_dsda_one_acc = np.zeros((n_tests, max_n_clips - 1))
        mAP_dssa_one_acc = np.zeros((n_tests, max_n_clips - 1))
        mAP_sssa_one_acc = np.zeros((n_tests, max_n_clips - 1))
        acc_dsda_all_vis = np.zeros((n_tests, max_n_clips - 1))
        acc_dssa_all_vis = np.zeros((n_tests, max_n_clips - 1))
        acc_sssa_all_vis = np.zeros((n_tests, max_n_clips - 1))

        for ns in tqdm(range(2, max_n_clips + 1)):
            # Matrices of distances including the right acceleration and accelerations from "i" more subjects
            dist_dsda_one_vid = np.zeros((n_ft, ns + 1))  # + 1 because it includes the matching acceleration as well
            dist_dssa_one_vid = np.zeros((n_ft, ns + 1))
            dist_sssa_one_vid = np.zeros((n_ft, ns + 1))
            dist_dsda_one_acc = np.zeros((n_ft, ns + 1))
            dist_dssa_one_acc = np.zeros((n_ft, ns + 1))
            dist_sssa_one_acc = np.zeros((n_ft, ns + 1))
            # For the "all_vis" case, all the ns subjects have an accelerometer and are visible from the camera. Store
            # the distance between each video and each acceleration. For ns subjects, the result is ns*ns distances
            dist_dsda_all_vis = np.zeros((n_ft, ns, ns))
            dist_dssa_all_vis = np.zeros((n_ft, ns, ns))
            dist_sssa_all_vis = np.zeros((n_ft, ns, ns))

            # The first element is the distance of the matching acceleration, which is on the diagonal of the distance
            # matrix
            dist_dsda_one_vid[:, 0] = np.diag(dist)
            dist_dssa_one_vid[:, 0] = np.diag(dist)
            dist_sssa_one_vid[:, 0] = np.diag(dist)
            dist_dsda_one_acc[:, 0] = np.diag(dist)
            dist_dssa_one_acc[:, 0] = np.diag(dist)
            dist_sssa_one_acc[:, 0] = np.diag(dist)

            # The remaining elements are "ns" different clips selected from different negative types.
            # To select the negative, a mask for the distance matrix is generated. This mask reflects the type of
            # negative which is chosen. Then, random accelerations are picked from the non-masked elements of the
            # distance matrix.
            mask_act = np.repeat(all_labels[None, :, 0], all_labels.shape[0], 0)
            mask_sub = np.repeat(all_labels[None, :, 1], all_labels.shape[0], 0)
            mask_dsda = np.logical_and(np.not_equal(mask_act, all_labels[:, 0, None]),
                                       np.not_equal(mask_sub, all_labels[:, 1, None]))
            mask_dssa = np.logical_and(np.equal(mask_act, all_labels[:, 0, None]),
                                       np.not_equal(mask_sub, all_labels[:, 1, None]))
            mask_sssa = np.logical_and(np.equal(mask_act, all_labels[:, 0, None]),
                                       np.equal(mask_sub, all_labels[:, 1, None]))
            for it in range(n_tests):
                for i in range(mask_dsda.shape[0]):
                    choices = np.random.choice(np.where(mask_dsda[i, :])[0], ns)
                    # For the single vid/acc cases, simply pick the distances between the ith video (acceleration) and the
                    # remaining accelerations (videos)
                    dist_dsda_one_vid[i, 1:] = dist[i, choices]
                    dist_dsda_one_acc[i, 1:] = dist[choices, i]
                    # For "all_vis" store all the distances between all the combinations of the ns videos and accelerations
                    ii = np.kron(choices, np.ones(ns)).astype('int')
                    jj = np.kron(np.ones(ns), choices).astype('int')
                    dist_dsda_all_vis[i, :, :] = dist[ii, jj].reshape((ns, ns))

                    choices = np.random.choice(np.where(mask_dssa[i, :])[0], ns)
                    dist_dssa_one_vid[i, 1:] = dist[i, choices]
                    dist_dssa_one_acc[i, 1:] = dist[choices, i]
                    ii = np.kron(choices, np.ones(ns)).astype('int')
                    jj = np.kron(np.ones(ns), choices).astype('int')
                    dist_dssa_all_vis[i, :, :] = dist[ii, jj].reshape((ns, ns))

                    choices = np.random.choice(np.where(mask_sssa[i, :])[0], ns)
                    dist_sssa_one_vid[i, 1:] = dist[i, choices]
                    dist_sssa_one_acc[i, 1:] = dist[choices, i]
                    ii = np.kron(choices, np.ones(ns)).astype('int')
                    jj = np.kron(np.ones(ns), choices).astype('int')
                    dist_sssa_all_vis[i, :, :] = dist[ii, jj].reshape((ns, ns))

                # The mAP for the single vid/acc cases are calculated by retrieving the order of the correct video (
                # acceleration) given the distances
                order = np.argsort(dist_dsda_one_vid, axis=1)
                mAP_dsda_one_vid[it, ns - 2] = np.mean(1 / (order[:, 0] + 1))  # - 2 because the first number of subjects
                                                                               # tested is 2
                order = np.argsort(dist_dsda_one_acc, axis=1)
                mAP_dsda_one_acc[it, ns - 2] = np.mean(1 / (order[:, 0] + 1))
                # For the "all_vis" case we measure the accuracy by considering the minimum distance of each
                # mini-distance matrix and comparing it to the correct position
                order = np.argmin(dist_dsda_all_vis, axis=2)
                acc_dsda_all_vis[it, ns - 2] = np.mean(np.equal(order, np.arange(ns)[None, ]))

                order = np.argsort(dist_dssa_one_vid, axis=1)
                mAP_dssa_one_vid[it, ns - 2] = np.mean(1 / (order[:, 0] + 1))
                order = np.argsort(dist_dssa_one_acc, axis=1)
                mAP_dssa_one_acc[it, ns - 2] = np.mean(1 / (order[:, 0] + 1))
                order = np.argmin(dist_dssa_all_vis, axis=2)
                acc_dssa_all_vis[it, ns - 2] = np.mean(np.equal(order, np.arange(ns)[None, ]))

                order = np.argsort(dist_sssa_one_vid, axis=1)
                mAP_sssa_one_vid[it, ns - 2] = np.mean(1 / (order[:, 0] + 1))
                order = np.argsort(dist_sssa_one_acc, axis=1)
                mAP_sssa_one_acc[it, ns - 2] = np.mean(1 / (order[:, 0] + 1))
                order = np.argmin(dist_sssa_all_vis, axis=2)
                acc_sssa_all_vis[it, ns - 2] = np.mean(np.equal(order, np.arange(ns)[None, ]))

        plt.figure(fig1.number)
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        markers = ['s', 'o', 'x', '^']

        name_dsda = '{} @ {}'.format(method_names[mi], 'DSDA') if method_names else 'DSDA'
        name_dssa = '{} @ {}'.format(method_names[mi], 'DSSA') if method_names else 'DSSA'
        name_sssa = '{} @ {}'.format(method_names[mi], 'SSSA') if method_names else 'SSSA'
        plt.plot(np.arange(2, max_n_clips +  1), np.mean(mAP_dsda_one_vid, axis=0), label=name_dsda, color=colors[mi], marker=markers[0])
        plt.plot(np.arange(2, max_n_clips + 1), np.mean(mAP_dssa_one_vid, axis=0), label=name_dssa, color=colors[mi], marker=markers[1])
        plt.plot(np.arange(2, max_n_clips + 1), np.mean(mAP_sssa_one_vid, axis=0), label=name_sssa, color=colors[mi], marker=markers[2])
        if mi == n_methods - 1:
            plt.plot(np.arange(2, max_n_clips + 1), 1/np.arange(2, max_n_clips + 1), '--k', label='Random')
        # plt.ylim([0, 1])
        plt.xlabel('#N hidden subjects with wearable')
        plt.ylabel('mAP')
        plt.title('One person in front of the camera')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(filepath, 'map_same_vid_vs_nclips.pdf'), bbox_inches='tight', pad_inches=0)

        plt.figure(fig2.number)
        plt.plot(np.arange(2, max_n_clips +  1), np.mean(mAP_dsda_one_acc, axis=0), label=name_dsda, color=colors[mi], marker=markers[0])
        plt.plot(np.arange(2, max_n_clips + 1), np.mean(mAP_dssa_one_acc, axis=0), label=name_dssa, color=colors[mi], marker=markers[1])
        plt.plot(np.arange(2, max_n_clips + 1), np.mean(mAP_sssa_one_acc, axis=0), label=name_sssa, color=colors[mi], marker=markers[2])
        if mi == n_methods - 1:
            plt.plot(np.arange(2, max_n_clips + 1), 1/np.arange(2, max_n_clips + 1), '--k', label='Random')
        # plt.ylim([0, 1])
        plt.xlabel('#N visible subjects without wearable')
        plt.ylabel('mAP')
        plt.title('One person with wearable')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(filepath, 'map_same_acc_vs_nclips.pdf'), bbox_inches='tight', pad_inches=0)

        plt.figure(fig3.number)
        plt.plot(np.arange(2, max_n_clips +  1), np.mean(acc_dsda_all_vis, axis=0), label=name_dsda, color=colors[mi], marker=markers[0])
        plt.plot(np.arange(2, max_n_clips + 1), np.mean(acc_dssa_all_vis, axis=0), label=name_dssa, color=colors[mi], marker=markers[1])
        plt.plot(np.arange(2, max_n_clips + 1), np.mean(acc_sssa_all_vis, axis=0), label=name_sssa, color=colors[mi], marker=markers[2])
        if mi == n_methods - 1:
            plt.plot(np.arange(2, max_n_clips + 1), 1/np.arange(2, max_n_clips + 1), '--k', label='Random')
        # plt.ylim([0, 1])
        plt.xlabel('#N visible subjects with wearable')
        plt.ylabel('Accuracy')
        plt.title('All subjects visible with wearable')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(filepath, 'acc_diff_vs_nclips.pdf'), bbox_inches='tight', pad_inches=0)

        print('{} assignment accuracy DSDA: {}'.format(method_names[mi], np.mean(acc_dsda_all_vis, axis=0)[0]))

    return fig1, fig2, fig3


def assignment_accuracy(vidbox_ft, acc_ft, all_labels, metric=euclidean):
    # Pretend that N people are in front of the camera with an accelerometer and match them with a greedy algorithm.
    n_tests = 1000
    n_max_people = 30
    acc = np.zeros((n_max_people))
    for i, n_people in enumerate(range(1, n_max_people + 1)):
        for test in range(n_tests):
            bf = np.random.permutation(acc_ft.shape[0])
            y_vid = vidbox_ft[bf[:n_people], :]
            y_acc = acc_ft[bf[:n_people], :]
            dist = cdist(y_vid, y_acc, 'euclidean')
            # dist = cdist(y_vid, y_acc, 'cosine')
            # dist = -cdist(y_vid, y_acc, lambda u,v: np.cov(u,v)[0,1])
            # dist = -cdist(y_vid, y_acc, lambda u,v: mutual_info_score(u,v))
            acc[i] += np.mean(np.equal(np.argmin(dist, axis=1), np. arange(0, n_people)))

        acc[i] /= n_tests

    plt.plot(range(1, n_max_people+1), acc)


def profile_comparison(vidbox_ft, acc_ft, all_labels, validgen, metric=euclidean, thr=None):
    keep_bs = validgen.batch_size
    validgen.batch_size = 1

    subj_target = 10
    subj_comp = 9
    # Only use the first session. Labels are progressive (1, 2, ...). At the end of the first session they go back to
    # 0. Use the difference between labels to select the first session
    lab_targ = all_labels[all_labels[:, 1] == subj_target]
    lab_comp = all_labels[all_labels[:, 1] == subj_comp]
    last_element = np.where(np.diff(lab_targ[:,0]) < 0)[0][0]
    lab_targ = lab_targ[:last_element]
    lab_comp = lab_comp[:last_element]
    vid_targ = vidbox_ft[all_labels[:, 1] == subj_target][:last_element]
    acc_targ = acc_ft[all_labels[:, 1] == subj_target][:last_element]
    vid_comp = vidbox_ft[all_labels[:, 1] == subj_comp][:last_element]
    acc_comp = acc_ft[all_labels[:, 1] == subj_comp][:last_element]

    # Calculate the score
    y_score_targ = np.array([metric(vid_targ[bf, :], acc_targ[bf, :]) for bf in range(acc_targ.shape[0])])
    y_score_comp = np.array([metric(vid_comp[bf, :], acc_targ[bf, :]) for bf in range(acc_targ.shape[0])])

    # Select a silhouette image for some frames, to show as samples
    n_frames = 23
    frame_ids = np.linspace(0, last_element, n_frames).astype('int')
    frames_targ = []
    rawacc_targ = []
    for frame_id in frame_ids:
        clips_db_i = [i for i, bf in enumerate(validgen.clips_db) if bf['name'] == 'Subject{}'.format(subj_target)][0]
        batch_id = [i for i, bf in enumerate(validgen.mapping['anchor'])
                        if bf[0] == clips_db_i and bf[1] == frame_id][0]
        data, _ = validgen.__getitem__(batch_id)
        vid, _, pacc, _ = data
        frames_targ.append(vid[0, ..., 0])
        rawacc_targ.append(pacc[0, ])

    rawacc_targ = np.vstack(rawacc_targ)

    frame_ids = np.where(all_labels[:, 1] == subj_comp)[0][np.linspace(0, last_element, n_frames).astype('int')]
    frame_ids -= frame_ids[0]
    frames_comp = []
    acc_comp = []
    for frame_id in frame_ids:
        clips_db_i = [i for i, bf in enumerate(validgen.clips_db) if bf['name'] == 'Subject{}'.format(subj_comp)][0]
        batch_id = [i for i, bf in enumerate(validgen.mapping['anchor'])
                        if bf[0] == clips_db_i and bf[1] == frame_id][0]
        data, _ = validgen.__getitem__(batch_id)
        vid, _, pacc, _ = data
        frames_comp.append(vid[0, ..., 0])
        acc_comp.append(pacc[0, ])

    acc_comp = np.vstack(acc_comp)

    plt.figure(figsize=(15, 4))
    gs = gridspec.GridSpec(5, 1, height_ratios=[2, 1.5, 2, 4, 1], hspace=0.5)
    ax = [plt.subplot(bf) for bf in gs]


    ax[0].imshow(np.hstack(frames_targ), cmap='gray')
    ax[0].set_title('Monitored subject video')
    ax[0].axis('off')

    ax[1].plot(rawacc_targ, clip_on=False)
    ax[1].set_xlim([0, len(rawacc_targ)])
    ax[1].set_title('Monitored subject accelerometer')
    ax[1].axis('off')

    ax[2].imshow(np.hstack(frames_comp), cmap='gray')
    ax[2].set_title('Guest video')
    ax[2].axis('off')

    ax[3].plot(y_score_targ, label='Matching distance')
    ax[3].plot(y_score_comp, label='Non matching distance')
    ax[3].set_xticks([])
    ymin = np.minimum(y_score_comp.min(), y_score_targ.min())
    ymax = np.maximum(y_score_comp.max(), y_score_targ.max())

    x_int = np.linspace(0, len(y_score_targ), 100000)
    x = np.arange(len(y_score_targ))
    y_comp_int = np.interp(x_int, x, y_score_comp)
    y_targ_int = np.interp(x_int, x, y_score_targ)

    if thr:
        match = np.logical_and(np.less(y_targ_int, thr), np.greater(y_comp_int, thr))
        ax[3].fill_between(x_int, ymin, ymax, where=(1-match), facecolor=[1, .8, .8])
        ax[3].fill_between(x_int, ymin, ymax, where=match, facecolor=[.6,1,.6])
        ax[3].axhline(thr, color='k', label='Threshold')
    else:
        ax[3].fill_between(x_int, ymin, ymax, where=np.less(y_comp_int, y_targ_int), facecolor=[1,.8,.8])
        ax[3].fill_between(x_int, ymin, ymax, where=np.greater_equal(y_comp_int, y_targ_int), facecolor=[.6,1,.6])
    ax[3].legend()
    ax[3].set_xlim([x_int[0], x_int[-1]])
    ax[3].set_xlabel('Time')
    ax[3].set_ylabel('Distance')

    labels = ('none', 'standing', 'sitting', 'walking', 'wiping', 'vacuuming', 'sweeping', 'lying',
              'exercising', 'stretching', 'cleaning', 'reading')

    lab_id = np.unique(all_labels[:, 0]).astype('int')

    for i, lab in enumerate(lab_id):
        where = np.where(lab_targ[:, 0] == lab)[0]
        ax[4].text(where[len(where)//2], 0.5, labels[lab], horizontalalignment='center', verticalalignment='center')
        ax[4].fill_between(x, -1, 2, where=lab_targ[:, 0] == lab, facecolor=[.8,.9,1])

    ax[4].set_xlim([x_int[0], x_int[-1]])
    ax[4].axis('off')
    # plt.tight_layout()
    plt.savefig('sample_results.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('sample_results.png', bbox_inches='tight', pad_inches=0)

    validgen.batch_size = keep_bs


def calculate_auROC_vs_n_people(vidbox_ft, acc_ft, all_labels, metric=euclidean):
    max_n_clips = 100
    au_random = np.zeros((max_n_clips))

    for ns in tqdm(range(2, max_n_clips)):
        # ROC with random negatives
        y_vid = np.tile(vidbox_ft, [ns+1, 1])
        bf = np.array([np.random.permutation(acc_ft.shape[0]) for _ in range(ns)]).flatten()
        y_acc = np.vstack((acc_ft, acc_ft[bf, :]))
        y_score = np.array([metric(y_vid[bf, :], y_acc[bf, :]) for bf in range(y_vid.shape[0])])  # np.sqrt(np.sum(np.square(y_vid - y_acc), axis=1))
        y_true = np.vstack((np.ones((vidbox_ft.shape[0], 1)), np.zeros((acc_ft.shape[0]*ns, 1))))
        au_random[ns] = roc_auc_score(y_true, -y_score)


def calculate_ROC(vidbox_ft, acc_ft, all_labels, validgen, metric=euclidean, filepath='', display=True, fig_name='', show_threshold=False):
    if display:
        plt.figure(figsize=(4,4))

    # ROC with random negatives
    y_vid = np.vstack((vidbox_ft, vidbox_ft))
    bf = np.random.permutation(acc_ft.shape[0])
    y_acc = np.vstack((acc_ft, acc_ft[bf, :]))
    y_score = np.array([metric(y_vid[bf, :], y_acc[bf, :]) for bf in range(y_vid.shape[0])])  # np.sqrt(np.sum(np.square(y_vid - y_acc), axis=1))
    y_true = np.vstack((np.ones((vidbox_ft.shape[0], 1)),
                        np.zeros((acc_ft.shape[0], 1))))
    fpr, tpr, thresholds = roc_curve(y_true, -y_score)
    auc_rand = roc_auc_score(y_true, -y_score)
    fnr = 1 - tpr
    eer = fnr[np.argmin(np.abs(fnr - fpr))]
    print('Random negative {:.3f}'.format(eer))

    # ROC with random negatives
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

    y_score = np.array([metric(y_vid[bf, :], y_acc[bf, :]) for bf in range(y_vid.shape[0])])  # np.sqrt(np.sum(np.square(y_vid - y_acc), axis=1))
    y_true = np.vstack((np.ones((vidbox_ft.shape[0], 1)),
                        np.zeros((acc_ft.shape[0], 1))))
    fpr, tpr, thresholds = roc_curve(y_true, -y_score)
    auc_dsda = roc_auc_score(y_true, -y_score)
    fnr = 1 - tpr
    eer = fnr[np.argmin(np.abs(fnr - fpr))]
    print('DSDA {:.3f}'.format(eer))
    if display:
        plt.plot(fpr, tpr, label='DSDA ({:.3f})'.format(auc_dsda))

    # Calculate the threshold
    if show_threshold:
        # Distance with the point (0, 1) 
        dist = np.square(fpr-0) + np.square(tpr-1)
        mdist = np.argmin(dist)
        fpr_opt = fpr[mdist]
        tpr_opt = tpr[mdist]

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

    y_score = np.array([metric(y_vid[bf, :], y_acc[bf, :]) for bf in range(y_vid.shape[0])])  # np.sqrt(np.sum(np.square(y_vid - y_acc), axis=1))
    y_true = np.vstack((np.ones((vidbox_ft.shape[0], 1)),
                        np.zeros((acc_ft.shape[0], 1))))
    fpr, tpr, thresholds = roc_curve(y_true, -y_score)
    auc_dssa = roc_auc_score(y_true, -y_score)
    fnr = 1 - tpr
    eer = fnr[np.argmin(np.abs(fnr - fpr))]
    print('DSSA {:.3f}'.format(eer))
    if display:
        plt.plot(fpr, tpr, label='DSSA ({:.3f})'.format(auc_dssa))

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

    y_score = np.array([metric(y_vid[bf, :], y_acc[bf, :]) for bf in range(y_vid.shape[0])])  # np.sqrt(np.sum(np.square(y_vid - y_acc), axis=1))
    y_true = np.vstack((np.ones((vidbox_ft.shape[0], 1)),
                        np.zeros((acc_ft.shape[0], 1))))
    fpr, tpr, thresholds = roc_curve(y_true, -y_score)
    auc_sssa = roc_auc_score(y_true, -y_score)
    fnr = 1 - tpr
    eer = fnr[np.argmin(np.abs(fnr - fpr))]
    print('SSSA {:.3f}'.format(eer))
    if display:
        plt.plot(fpr, tpr, label='SSSA ({:.3f})'.format(auc_sssa))

    # ROC with ADJ
    bf = np.argsort(validgen.mapping_permutation)
    y_vid = np.vstack((vidbox_ft[bf], vidbox_ft[bf]))
    y_acc = np.vstack((acc_ft[bf], acc_ft[bf][np.arange(len(acc_ft)) - 1]))
    y_score = np.array([metric(y_vid[bf, :], y_acc[bf, :]) for bf in range(y_vid.shape[0])])  # np.sqrt(np.sum(np.square(y_vid - y_acc), axis=1))
    y_true = np.vstack((np.ones((vidbox_ft.shape[0], 1)),
                        np.zeros((acc_ft.shape[0], 1))))
    fpr, tpr, thresholds = roc_curve(y_true, -y_score)
    auc_prev = roc_auc_score(y_true, -y_score)
    fnr = 1 - tpr
    eer = fnr[np.argmin(np.abs(fnr - fpr))]
    print('Previous clip {:.3f}'.format(eer))
    
    if display:
        plt.plot(fpr, tpr, label='OVLP ({:.3f})'.format(auc_prev))

        plt.plot([0, 1], [0, 1], '--k')
        
        if show_threshold:
            plt.plot([0, fpr_opt], [1, tpr_opt], ':', color=[0.6, 0.6, 0.6], label='Optimal threshold')

        plt.legend(loc='lower right')
        plt.axis('square')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])

        plt.savefig(os.path.join(filepath, 'ROC_curves_' + fig_name + '.pdf'), bbox_inches='tight', pad_inches=0)

    return auc_rand, auc_prev, auc_dsda, auc_dssa, auc_sssa

def table_auroc_results_from_txt(results_folder='',
                                 cabrera_vid=None, cabrera_acc=None, cabrera_labels=None,
                                 shigeta_vid=None, shigeta_acc=None, shigeta_labels=None, validgen=None, filepath=''):
    # %%
    # Define the possible experiments
    strategy_names = ['Easy', 'Easy/Hard', 'Hard', 'Hard/VeryH', 'VeryH', 'All']
    strategies = ['50_DSDA-50_DSSA', '25_DSDA-25_DSSA-50_SSSA', '100_SSSA', '50_SSSA-50_ADJ', '100_ADJ',
                          '11_DSDA-11_DSSA-11_SSDA-33_SSSA-33_ADJ']
    models = ['fully_conv', 'reid_pooling_v1', 'reid_nopool_v1']
    losses = ['triplet_loss', 'my_triplet_loss']

    # Load the data in dictionaries
    au_prev = {}
    au_dsda = {}
    au_dssa = {}
    au_sssa = {}
    au_avg = {}
    for model in models:
        plt.figure('auroc {}'.format(model))
        au_prev[model] = {}
        au_dsda[model] = {}
        au_dssa[model] = {}
        au_sssa[model] = {}
        au_avg[model] = {}
        for strategy in strategies:
            au_prev[model][strategy] = {}
            au_dsda[model][strategy] = {}
            au_dssa[model][strategy] = {}
            au_sssa[model][strategy] = {}
            au_avg[model][strategy] = {}
            for loss in losses:
                exp_name = 'xfin_{}_{}_{}'.format(strategy, model, loss)
                exp_dir = os.path.join(results_folder, exp_name)
                try:
                    json_file = os.path.join(exp_dir, 'auc_history.txt')
                    with open(json_file) as data_file:
                        data = json.load(data_file)
                        avg = (np.array(data['prev'])*100 +
                               np.array(data['dsda'])*100 +
                               np.array(data['dssa'])*100 +
                               np.array(data['sssa'])*100)/4
                        au_prev[model][strategy][loss] = np.array(data['prev'])[avg.argmax()]*100
                        au_dsda[model][strategy][loss] = np.array(data['dsda'])[avg.argmax()]*100
                        au_dssa[model][strategy][loss] = np.array(data['dssa'])[avg.argmax()]*100
                        au_sssa[model][strategy][loss] = np.array(data['sssa'])[avg.argmax()]*100
                        au_avg[model][strategy][loss] = avg.max()
                except IOError:
                    print('{} not found'.format(json_file))
                    au_prev[model][strategy][loss] = np.nan
                    au_dsda[model][strategy][loss] = np.nan
                    au_dssa[model][strategy][loss] = np.nan
                    au_sssa[model][strategy][loss] = np.nan
                    au_avg[model][strategy][loss] = np.nan

    # Generate a Latex table per model
    with open('table_template.txt', 'r') as f:
        template = f.read()

    for model in models:
        print('\n\n\tTable for {}:'.format(model))
        print(template.format(ee_dsda_stl=au_dsda[model][strategies[0]]['triplet_loss'],
                              ee_dssa_stl=au_dssa[model][strategies[0]]['triplet_loss'],
                              ee_sssa_stl=au_sssa[model][strategies[0]]['triplet_loss'],
                              ee_prev_stl=au_prev[model][strategies[0]]['triplet_loss'],
                              ee_avg_stl=au_avg[model][strategies[0]]['triplet_loss'],
                              ee_dsda_rtl=au_dsda[model][strategies[0]]['my_triplet_loss'],
                              ee_dssa_rtl=au_dssa[model][strategies[0]]['my_triplet_loss'],
                              ee_sssa_rtl=au_sssa[model][strategies[0]]['my_triplet_loss'],
                              ee_prev_rtl=au_prev[model][strategies[0]]['my_triplet_loss'],
                              ee_avg_rtl=au_avg[model][strategies[0]]['my_triplet_loss'],
                              eh_dsda_stl=au_dsda[model][strategies[1]]['triplet_loss'],
                              eh_dssa_stl=au_dssa[model][strategies[1]]['triplet_loss'],
                              eh_sssa_stl=au_sssa[model][strategies[1]]['triplet_loss'],
                              eh_prev_stl=au_prev[model][strategies[1]]['triplet_loss'],
                              eh_avg_stl=au_avg[model][strategies[1]]['triplet_loss'],
                              eh_dsda_rtl=au_dsda[model][strategies[1]]['my_triplet_loss'],
                              eh_dssa_rtl=au_dssa[model][strategies[1]]['my_triplet_loss'],
                              eh_sssa_rtl=au_sssa[model][strategies[1]]['my_triplet_loss'],
                              eh_prev_rtl=au_prev[model][strategies[1]]['my_triplet_loss'],
                              eh_avg_rtl=au_avg[model][strategies[1]]['my_triplet_loss'],
                              hh_dsda_stl=au_dsda[model][strategies[2]]['triplet_loss'],
                              hh_dssa_stl=au_dssa[model][strategies[2]]['triplet_loss'],
                              hh_sssa_stl=au_sssa[model][strategies[2]]['triplet_loss'],
                              hh_prev_stl=au_prev[model][strategies[2]]['triplet_loss'],
                              hh_avg_stl=au_avg[model][strategies[2]]['triplet_loss'],
                              hh_dsda_rtl=au_dsda[model][strategies[2]]['my_triplet_loss'],
                              hh_dssa_rtl=au_dssa[model][strategies[2]]['my_triplet_loss'],
                              hh_sssa_rtl=au_sssa[model][strategies[2]]['my_triplet_loss'],
                              hh_prev_rtl=au_prev[model][strategies[2]]['my_triplet_loss'],
                              hh_avg_rtl=au_avg[model][strategies[2]]['my_triplet_loss'],
                              hv_dsda_stl=au_dsda[model][strategies[3]]['triplet_loss'],
                              hv_dssa_stl=au_dssa[model][strategies[3]]['triplet_loss'],
                              hv_sssa_stl=au_sssa[model][strategies[3]]['triplet_loss'],
                              hv_prev_stl=au_prev[model][strategies[3]]['triplet_loss'],
                              hv_avg_stl=au_avg[model][strategies[3]]['triplet_loss'],
                              hv_dsda_rtl=au_dsda[model][strategies[3]]['my_triplet_loss'],
                              hv_dssa_rtl=au_dssa[model][strategies[3]]['my_triplet_loss'],
                              hv_sssa_rtl=au_sssa[model][strategies[3]]['my_triplet_loss'],
                              hv_prev_rtl=au_prev[model][strategies[3]]['my_triplet_loss'],
                              hv_avg_rtl=au_avg[model][strategies[3]]['my_triplet_loss'],
                              vh_dsda_stl=au_dsda[model][strategies[4]]['triplet_loss'],
                              vh_dssa_stl=au_dssa[model][strategies[4]]['triplet_loss'],
                              vh_sssa_stl=au_sssa[model][strategies[4]]['triplet_loss'],
                              vh_prev_stl=au_prev[model][strategies[4]]['triplet_loss'],
                              vh_avg_stl=au_avg[model][strategies[4]]['triplet_loss'],
                              vh_dsda_rtl=au_dsda[model][strategies[4]]['my_triplet_loss'],
                              vh_dssa_rtl=au_dssa[model][strategies[4]]['my_triplet_loss'],
                              vh_sssa_rtl=au_sssa[model][strategies[4]]['my_triplet_loss'],
                              vh_prev_rtl=au_prev[model][strategies[4]]['my_triplet_loss'],
                              vh_avg_rtl=au_avg[model][strategies[4]]['my_triplet_loss'],
                              aa_dsda_stl=au_dsda[model][strategies[5]]['triplet_loss'],
                              aa_dssa_stl=au_dssa[model][strategies[5]]['triplet_loss'],
                              aa_sssa_stl=au_sssa[model][strategies[5]]['triplet_loss'],
                              aa_prev_stl=au_prev[model][strategies[5]]['triplet_loss'],
                              aa_avg_stl=au_avg[model][strategies[5]]['triplet_loss'],
                              aa_dsda_rtl=au_dsda[model][strategies[5]]['my_triplet_loss'],
                              aa_dssa_rtl=au_dssa[model][strategies[5]]['my_triplet_loss'],
                              aa_sssa_rtl=au_sssa[model][strategies[5]]['my_triplet_loss'],
                              aa_prev_rtl=au_prev[model][strategies[5]]['my_triplet_loss'],
                              aa_avg_rtl=au_avg[model][strategies[5]]['my_triplet_loss']))

    if cabrera_vid is not None:
        auc_rand_cab, auc_prev_cab, auc_dsda_cab, auc_dssa_cab, auc_sssa_cab = \
            calculate_ROC(cabrera_vid, cabrera_acc, cabrera_labels, validgen, display=False)
        auc_rand_shi, auc_prev_shi, auc_dsda_shi, auc_dssa_shi, auc_sssa_shi = \
            calculate_ROC(shigeta_vid, shigeta_acc, shigeta_labels, validgen, display=False)

        with open('table_template_bl.txt', 'r') as f:
            template = f.read()

        print('\n\n\tTable for baseline:')
        print(template.format(shi_dsda=auc_dsda_shi*100,
                              shi_dssa=auc_dssa_shi*100,
                              shi_sssa=auc_sssa_shi*100,
                              shi_prev=auc_prev_shi*100,
                              shi_avg=(auc_dsda_shi + auc_dssa_shi + auc_sssa_shi + auc_prev_shi)/4*100,
                              cab_dsda=auc_dsda_cab*100,
                              cab_dssa=auc_dssa_cab*100,
                              cab_sssa=auc_sssa_cab*100,
                              cab_prev=auc_prev_cab*100,
                              cab_avg=(auc_dsda_cab + auc_dssa_cab + auc_sssa_cab + auc_prev_cab)/4*100))


def md5_file(fname):
    '''From https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file'''
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


if __name__ == '__main__':
    model_name = r"xfin_25_DSDA-25_DSSA-50_SSSA_fully_conv_my_triplet_loss\AUC_DSDA.h5"  # Best model
    data_type = 'acc_sil_100_00_clean'
    batch_size = 4

    cache_path = r'./cache'

    os.makedirs(cache_path, exist_ok=True)

    # Paths
    max_queue_size = 1
    data_path = os.path.join(r'./ReID', data_type)
    load_zip_memory = False
    use_multiprocess = False
    n_workers = 1

    model_architecture = 'fully_conv'
    # model_architecture = 'reid_nopool_v1'
    # model_architecture = 'reid_pooling_v1'

    model = triplet_network(model_architecture)
    model.load_weights(model_name)
    model_hash = md5_file(model_name)
    print(model_hash)

    acc_type = 'ACC1.000000'
    negative_type = '50_DSDA-50_DSSA'

    activities = ('walking', 'wiping', 'vacuuming', 'sweeping', 'exercising', 'stretching', 'cleaning')
    # activities = ('standing', 'sitting', 'walking', 'wiping', 'vacuuming', 'sweeping', 'exercising', 'stretching', 'cleaning', 'reading')

    cabrera_vid, cabrera_acc, cabrera_labels = cabrera_features(data_path, load_zip_memory, acc_type, activities,
                                                                subjects=range(8, 11), cache=cache_path)

    shigeta_vid, shigeta_acc, shigeta_labels = shigeta_features(data_path, load_zip_memory, acc_type, activities,
                                                                subjects=range(8, 11), cache=cache_path)

    vidbox_ft, acc_ft, all_labels, validgen = calculate_features(model, data_path, batch_size, load_zip_memory,
                                                                 acc_type, activities, subjects=range(8, 11),
                                                                 cache=cache_path, md5=model_hash, cache_mod=model_architecture)

    profile_comparison(vidbox_ft, acc_ft, all_labels, validgen)
    calculate_mAP(vidbox_ft, acc_ft, all_labels, validgen)
    calculate_ROC(vidbox_ft, acc_ft, all_labels, validgen, fig_name='ours', show_threshold=True)
    calculate_ROC(cabrera_vid, cabrera_acc, cabrera_labels, validgen, fig_name='cabrera')
    calculate_ROC(shigeta_vid, shigeta_acc, shigeta_labels, validgen, fig_name='shigeta')
    variable_clip_length([cabrera_vid, shigeta_vid, vidbox_ft], [cabrera_acc, shigeta_acc, acc_ft], all_labels,
                          method_names=['Cabrera et al.', 'Shigeta et al.', 'Proposed Method'])
    metrics_vs_n_people([cabrera_vid, shigeta_vid, vidbox_ft], [cabrera_acc, shigeta_acc, acc_ft], all_labels,
                        method_names=['Cabrera et al.', 'Shigeta et al.', 'Proposed Method'])

    table_auroc_results_from_txt(r'/xfin', cabrera_vid, cabrera_acc, cabrera_labels, shigeta_vid, shigeta_acc, shigeta_labels, validgen)