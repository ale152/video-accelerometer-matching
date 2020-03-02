import os
import zipfile
import cv2
import numpy as np

from io import BytesIO
from keras.utils import Sequence
from math import ceil
from random import seed
from PIL import Image
from tqdm import tqdm


class BatchGenerator(Sequence):
    def __init__(self, data_folder='', batch_size=4, video_size=(100, 100, 100), acc_size=100,
                 box_size=100, acc_folder='ACC0.000000', name='Generic', verbose=True, shuffle=False, random_seed=0,
                 data_augmentation=False, negative_type='dsda', load_zip_memory=False, subjects=None, filter=False,
                 activity=None, acc_augmentation=False, vid_augmentation=False):

        # Boring class stuff
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.name = name
        self.verbose = verbose
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        self.negative_type = negative_type
        self.acc_augmentation = acc_augmentation
        self.vid_augmentation = vid_augmentation
        self.activity = activity

        self.vid_h, self.vid_w, self.vid_t = video_size
        self.acc_t = acc_size
        self.box_t = box_size

        self.acc_folder = acc_folder
        self.vid_folder = 'better_silhouette'
        self.box_folder = 'bounding_boxes'

        self.mapping_permutation = None

        self.activities = ('none', 'standing', 'sitting', 'walking', 'wiping', 'vacuuming', 'sweeping', 'lying',
                           'exercising', 'stretching', 'cleaning', 'reading')

        # Set random seed
        seed(random_seed)
        np.random.seed(random_seed)

        # Deal with the activity filtering. Activity can either be a single string or a list of strings. Otherwise None
        if not isinstance(activity, list):
            activity = [activity]

        # List the zip paths and apply filter/limit
        self.zip_paths = [os.path.join(self.data_folder, bf)
                          for bf in os.listdir(self.data_folder)
                          if bf.endswith('.zip')]
        if filter:
            self.zip_paths = [bf for bf in self.zip_paths if filter in bf]
            print('Filtering by {}:\n{}'.format(filter, self.zip_paths))
        if subjects:
            self.zip_paths  = [bf for bf in self.zip_paths
                               if int(os.path.basename(bf).split('Subject')[1].split('_')[0]) in subjects]
            print('Filtering by subject {}:\n{}'.format(subjects, self.zip_paths))

        # Define two lists containing the data
        # self.zips_db = [ZipFile('file1'), ZipFile('file2'), ...]
        # self.clips_db = [{'name': 'Subject1',
        #                   'clips': ['clip_00000',
        #                             'clip_00001',
        #                              ...       ],
        #                   'zip_id': [0, 0, 1]}]
        self.zips_db = []
        self.clips_db = []

        # Fill the lists from the zip files
        for zip_i, zip_path in enumerate(tqdm(self.zip_paths, 'Reading zip files')):
            subject = os.path.basename(zip_path).split('_')[0]
            # Add subject to the list of subjects
            existing_subjects = [bf['name'] for bf in self.clips_db]
            if subject not in existing_subjects:
                self.clips_db.append({'name': subject, 'clips':[], 'zip_id': []})

            # Open the zip and read the content
            zip_file = self.load_zip(zip_path, load_in_memory=load_zip_memory)
            self.zips_db.append(zip_file)
            list_of_clips = [os.path.dirname(bf) for bf in zip_file.namelist() if bf.endswith('sil_0.png')]

            # Filter the list of images by activity
            if activity:
                list_of_clips = [clip for clip in list_of_clips if any([True for act in activity if act in clip])]
                if not list_of_clips:
                    print('\nWarning /!\\ {} does not contain images for "{}"'.format(zip_path, activity))

            zip_id = len(self.zips_db) - 1
            self.clips_db[-1]['clips'].extend(list_of_clips)
            self.clips_db[-1]['zip_id'].extend([zip_id for _ in range(len(list_of_clips))])

        # Generate mapping, a global list of clips that associates batch elements to clips
        self.mapping = {'anchor':[], 'negative':[]}
        self.generate_mapping()

    def generate_mapping(self, epoch=0, logs=None):
        # First, fill the mapping array with the anchors
        self.mapping['anchor'] = []
        self.mapping['negative'] = []
        for sub_i, subject in enumerate(self.clips_db):
            n = len(subject['clips'])
            self.mapping['anchor'].extend([[sub_i, bf] for bf in range(n)])

        for ii, (sub_i, cl_id) in enumerate(tqdm(self.mapping['anchor'], 'Generating mapping')):
            # Select an acceleration _not_ matching with the video clip. It will be taken from
            anchor_subject = self.clips_db[sub_i]['name']
            anchor_activity = self.clips_db[sub_i]['clips'][cl_id].split('/')[1]

            neg_sub, possible_clips = self._pick_negative(self.negative_type, anchor_subject,
                                                          anchor_activity, ii)

            assert len(possible_clips) > 0, 'No negative clips found for anchor {} {} using {}'.format(
                anchor_subject, anchor_activity, self.negative_type)
            random_choice = np.random.randint(0, len(possible_clips))
            neg_clip = self.clips_db[neg_sub]['clips'].index(possible_clips[random_choice])

            self.mapping['negative'].append([neg_sub, neg_clip])

        if self.shuffle:
            permutation = np.random.permutation(len(self.mapping['anchor']))
            self.mapping['anchor'] = [self.mapping['anchor'][bf] for bf in permutation]
            self.mapping['negative'] = [self.mapping['negative'][bf] for bf in permutation]
            self.mapping_permutation = permutation
        else:
            self.mapping_permutation = np.arange(len(self.mapping['anchor']))


    def load_zip(self, zip_file_name, load_in_memory):
        # Find all the videos in the archive
        self.log('Loading the archive %s...' % zip_file_name)
        if load_in_memory:
            # Load the entire zip file in memory
            chunk_size = 1000000  # 1 MB
            with open(zip_file_name, 'rb') as file_archive:
                # Calculate the file size and read it by chunks of 1 MB each
                file_archive.seek(0, os.SEEK_END)
                size = file_archive.tell()
                n_chunks = size//chunk_size + 1
                file_archive.seek(0)
                # Allocate the buffer in memory
                archive_content = BytesIO(b'\x00' * size)
                archive_content.seek(0)
                for _ in tqdm(range(n_chunks) ):
                    archive_content.write(file_archive.read(chunk_size))

            my_zip = zipfile.ZipFile(archive_content, 'r')
        else:
            # Load the zip file from the disk
            my_zip = zipfile.ZipFile(zip_file_name, 'r')

        return my_zip

    def log(self, text):
        if self.verbose:
            print('<%s> %s' % (self.name, text))

    def load_clip(self, map_elem):
        subj, cl_id = map_elem
        target_zip = self.zips_db[self.clips_db[subj]['zip_id'][cl_id]]
        target_file = self.clips_db[subj]['clips'][cl_id]
        video = np.zeros((self.vid_h, self.vid_w, self.vid_t))
        for i in range(self.vid_t):
            frame_file = target_file + '/sil_%d.png' % i
            file = target_zip.read(frame_file)
            image = np.array(Image.open(BytesIO(file)))
            video[..., i] = image[..., 0]/255

        if self.vid_augmentation:
            # Random dilation/erosion
            max_kernel = 5
            siz = 1 + np.random.randint(max_kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (siz, siz))
            if np.random.randint(2) == 1:
                for i in range(video.shape[-1]):
                    video[..., i] = cv2.dilate(video[..., i], kernel, iterations=1)
            else:
                for i in range(video.shape[-1]):
                    video[..., i] = cv2.erode(video[..., i], kernel, iterations=1)

            # Random flip
            if np.random.randint(2) == 1:
                video = video[:, ::-1, :]

            # Salt and pepper
            percent = 3
            perm = np.random.choice(video.size, int(video.size*percent/100))
            video[np.unravel_index(perm, video.shape)] = 1 - video[np.unravel_index(perm, video.shape)]

        return video

    def _pick_random_subject_excluding(self, subject_excluded):
        existing_subjects = [bf['name'] for bf in self.clips_db]
        possible_subjects = existing_subjects[:]
        possible_subjects.remove(subject_excluded)
        random_choice = np.random.randint(0, len(possible_subjects))
        negative_subject = possible_subjects[random_choice]
        subj = existing_subjects.index(negative_subject)
        return subj

    def load_acc(self, map_elem):
        subj, cl_id = map_elem
        target_zip = self.zips_db[self.clips_db[subj]['zip_id'][cl_id]]
        target_file = self.clips_db[subj]['clips'][cl_id]

        target_file = target_file.replace(self.vid_folder, self.acc_folder)
        acc_file = target_file + '/acc.csv'
        file_content = target_zip.read(acc_file)
        acc_data = np.loadtxt(BytesIO(file_content), delimiter=',')[:, 1:]

        acc_resampled = np.zeros((self.acc_t, 3))
        new_t = np.linspace(0, acc_data.shape[0], self.acc_t)
        old_t = np.arange(0, acc_data.shape[0])
        for i in range(3):
            acc_resampled[:, i] = np.interp(new_t, old_t, acc_data[:, i])

        acc_resampled -= np.mean(acc_resampled, axis=0)

        if self.acc_augmentation:
            random_perm = np.random.permutation(3)
            acc_resampled = acc_resampled[:, random_perm]

        return acc_resampled

    def load_box(self, map_elem):
        subj, cl_id = map_elem
        target_zip = self.zips_db[self.clips_db[subj]['zip_id'][cl_id]]
        target_file = self.clips_db[subj]['clips'][cl_id]

        target_file = target_file.replace(self.vid_folder, self.box_folder)
        box_file = target_file + '/boxes.csv'
        file_content = target_zip.read(box_file)
        box_data = np.loadtxt(BytesIO(file_content), delimiter=',')[:, -4:]

        box_data[:, 0] /= 640
        box_data[:, 1] /= 480
        box_data[:, 2] /= 640
        box_data[:, 3] /= 480

        box_data -= np.mean(box_data, axis=0)

        return box_data

    def __len__(self):
        return ceil(len(self.mapping['anchor']) / self.batch_size)

    def __getitem__(self, idx):
        '''Generate the batch of video data'''
        sel_range = [idx * self.batch_size, (idx + 1) * self.batch_size]

        assert sel_range[0] < len(self.mapping['anchor']), 'The item selected ({}) exceeds the number of items ' \
                                                           'available ({}). Batch requested {} of {}'.format(
                                                            sel_range[0],
                                                            len(self.mapping['anchor']),
                                                            idx,
                                                            self.__len__())

        # Deal with last batch
        if sel_range[1] > len(self.mapping['anchor']):
            sel_range[1] = len(self.mapping['anchor'])
        real_batch_size = (sel_range[1] - sel_range[0])

        # Select the elements from the mapping array
        selected = [bf for bf in range(sel_range[0], sel_range[1])]

        # Initialise the batch
        batch_anchor_vid = np.zeros((real_batch_size, self.vid_h, self.vid_w, self.vid_t), dtype=np.float32)
        batch_anchor_box = np.zeros((real_batch_size, self.box_t, 4), dtype=np.float32)
        batch_positive_acc = np.zeros((real_batch_size, self.acc_t, 3), dtype=np.float32)
        batch_negative_acc = np.zeros((real_batch_size, self.acc_t, 3), dtype=np.float32)

        batch_labels = np.zeros((real_batch_size, 2))
        for batch_i in range(real_batch_size):
            # Load anchor
            batch_anchor_vid[batch_i, ...] = self.load_clip(self.mapping['anchor'][selected[batch_i]])
            batch_anchor_box[batch_i, ...] = self.load_box(self.mapping['anchor'][selected[batch_i]])

            batch_negative_acc[batch_i, ...] = self.load_acc(self.mapping['negative'][selected[batch_i]])
            batch_positive_acc[batch_i, ...] = self.load_acc(self.mapping['anchor'][selected[batch_i]])

            subj, cl_id = self.mapping['anchor'][selected[batch_i]]
            activity = self.clips_db[subj]['clips'][cl_id]
            activity = activity.split('/')[-2]
            batch_labels[batch_i, 0] = self.activities.index(activity)
            batch_labels[batch_i, 1] = int(self.clips_db[subj]['name'].split('Subject')[1])

        # Model([input_vid, input_box, input_acc_pos, input_acc_neg], distances)
        return [batch_anchor_vid, batch_anchor_box, batch_positive_acc, batch_negative_acc], batch_labels

    def _DSDA(self, negative_type, anchor_subject, anchor_activity, ii):
        '''(D)ifferent (S)ubject and (D)ifferent (A)ctivity'''
        possible_clips = []
        # Not all the subjects have all the activities, therefore loop until a random subject has this_activity
        while len(possible_clips) == 0:
            neg_sub = self._pick_random_subject_excluding(anchor_subject)
            possible_clips = [bf for bf in self.clips_db[neg_sub]['clips'] if anchor_activity not in bf]
        return neg_sub, possible_clips

    def _DSAA(self, negative_type, anchor_subject, anchor_activity, ii):
        '''(D)ifferent (S)ubject and (A)ny (A)ctivity'''
        neg_sub = self._pick_random_subject_excluding(anchor_subject)
        possible_clips = [bf for bf in self.clips_db[neg_sub]['clips']]
        return neg_sub, possible_clips

    def _DSSA(self, negative_type, anchor_subject, anchor_activity, ii):
        '''(D)ifferent (S)ubject from (S)ame (A)ctivity'''
        possible_clips = []
        # Not all the subjects have all the activities, therefore loop until a random subject has this_activity
        while len(possible_clips) == 0:
            neg_sub = self._pick_random_subject_excluding(anchor_subject)
            possible_clips = [bf for bf in self.clips_db[neg_sub]['clips'] if anchor_activity in bf]
        return neg_sub, possible_clips

    def _ASSA(self, negative_type, anchor_subject, anchor_activity, ii):
        '''(A)ny (S)ubject from (S)ame (A)ctivity'''
        possible_clips = []
        # Not all the subjects have all the activities, therefore loop until a random subject has this_activity
        while len(possible_clips) == 0:
            neg_sub = np.random.randint(0, len(self.clips_db))
            possible_clips = [bf for bf in self.clips_db[neg_sub]['clips'] if anchor_activity in bf]
        return neg_sub, possible_clips

    def _ASAA(self, negative_type, anchor_subject, anchor_activity, ii):
        '''(A)ny (S)ubject and (A)ny (A)ctivity'''
        neg_sub = np.random.randint(0, len(self.clips_db))
        possible_clips = [bf for bf in self.clips_db[neg_sub]['clips']]
        return neg_sub, possible_clips

    def _SSSA(self, negative_type, anchor_subject, anchor_activity, ii):
        '''(S)ame (S)ubject from (S)ame (A)ctivity'''
        neg_sub, _ = self.mapping['anchor'][ii]
        possible_clips = [bf for bf in self.clips_db[neg_sub]['clips'] if anchor_activity in bf]
        return neg_sub, possible_clips

    def _SSDA(self, negative_type, anchor_subject, anchor_activity, ii):
        '''(S)ame (S)ubject from (D)ifferent (A)ctivity'''
        neg_sub, _ = self.mapping['anchor'][ii]
        possible_clips = [bf for bf in self.clips_db[neg_sub]['clips'] if anchor_activity not in bf]
        return neg_sub, possible_clips

    def _ADJ(self, negative_type, anchor_subject, anchor_activity, ii):
        # Use adjacent clips as negative. Randomly pick between N clips before and N clips after
        neg_sub, cl_id = self.mapping['anchor'][ii]
        n_adj = 10
        i1 = cl_id - n_adj if cl_id - n_adj > 0 else 0
        i2 = cl_id + n_adj + 1
        possible_clips = self.clips_db[neg_sub]['clips'][i1:i2]
        possible_clips.remove(self.clips_db[neg_sub ]['clips'][cl_id])
        return neg_sub, possible_clips

    def _pick_negative(self, negative_type, anchor_subject, anchor_activity, ii):
        if negative_type == 'dsda':
            neg_sub, possible_clips = self._DSDA(negative_type, anchor_subject, anchor_activity, ii)
        elif negative_type == 'dsaa':
            neg_sub, possible_clips = self._DSAA(negative_type, anchor_subject, anchor_activity, ii)
        elif negative_type == 'dssa':
            neg_sub, possible_clips = self._DSSA(negative_type, anchor_subject, anchor_activity, ii)
        elif negative_type == 'assa':
            neg_sub, possible_clips = self._ASSA(negative_type, anchor_subject, anchor_activity, ii)
        elif negative_type == 'asaa':
            neg_sub, possible_clips = self._ASAA(negative_type, anchor_subject, anchor_activity, ii)
        elif negative_type == 'sssa':
            neg_sub, possible_clips = self._SSSA(negative_type, anchor_subject, anchor_activity, ii)
        elif negative_type == 'ssda':
            neg_sub, possible_clips = self._SSDA(negative_type, anchor_subject, anchor_activity, ii)
        elif negative_type == 'adj':
            neg_sub, possible_clips = self._ADJ(negative_type, anchor_subject, anchor_activity, ii)
        elif negative_type == 'alternate':
            # Alternate between DSSA, SSDA and SSSA
            if ii % 3 == 0:
                neg_sub, possible_clips = self._DSSA(negative_type, anchor_subject, anchor_activity, ii)
            elif ii % 3 == 1:
                neg_sub, possible_clips = self._SSDA(negative_type, anchor_subject, anchor_activity, ii)
            else:
                neg_sub, possible_clips = self._SSSA(negative_type, anchor_subject, anchor_activity, ii)
        elif negative_type == '33':
            # Alternate between DSDA, SSSA and ADJ
            if ii % 3 == 0:
                neg_sub, possible_clips = self._DSDA(negative_type, anchor_subject, anchor_activity, ii)
            elif ii % 3 == 1:
                neg_sub, possible_clips = self._SSSA(negative_type, anchor_subject, anchor_activity, ii)
            else:
                neg_sub, possible_clips = self._ADJ(negative_type, anchor_subject, anchor_activity, ii)
        elif negative_type == '75-25':
            # Alternate between 75% DSDA, 25% SSSA
            if ii % 4 < 3:
                neg_sub, possible_clips = self._DSDA(negative_type, anchor_subject, anchor_activity, ii)
            else:
                neg_sub, possible_clips = self._SSSA(negative_type, anchor_subject, anchor_activity, ii)
        elif negative_type == '50_DSDA-50_DSSA':
            # Alternate between 50% DSDA, 50% DSSA
            if ii % 2 == 0:
                neg_sub, possible_clips = self._DSDA(negative_type, anchor_subject, anchor_activity, ii)
            else:
                neg_sub, possible_clips = self._DSSA(negative_type, anchor_subject, anchor_activity, ii)
        elif negative_type == '25_DSDA-25_DSSA-50_SSSA':
            if ii % 2 == 0:
                if ii % 4 == 0:
                    neg_sub, possible_clips = self._DSDA(negative_type, anchor_subject, anchor_activity, ii)
                if ii % 4 == 2:
                    neg_sub, possible_clips = self._DSSA(negative_type, anchor_subject, anchor_activity, ii)
            else:
                neg_sub, possible_clips = self._SSSA(negative_type, anchor_subject, anchor_activity, ii)
        elif negative_type == '100_SSSA':
            neg_sub, possible_clips = self._SSSA(negative_type, anchor_subject, anchor_activity, ii)
        elif negative_type == '50_SSSA-50_ADJ':
            # Alternate between 50% DSDA, 50% ADJ
            if ii % 2 == 0:
                neg_sub, possible_clips = self._SSSA(negative_type, anchor_subject, anchor_activity, ii)
            else:
                neg_sub, possible_clips = self._ADJ(negative_type, anchor_subject, anchor_activity, ii)
        elif negative_type == '100_ADJ':
            neg_sub, possible_clips = self._ADJ(negative_type, anchor_subject, anchor_activity, ii)
        elif negative_type == '11_DSDA-11_DSSA-11_SSDA-33_SSSA-33_ADJ':
            # Alternate between 33% DSDA, 33% DSSA and 33% SSSA
            if ii % 3 == 0:
                if ii % 9 == 0:
                    neg_sub, possible_clips = self._DSDA(negative_type, anchor_subject, anchor_activity, ii)
                elif ii % 9 == 3:
                    neg_sub, possible_clips = self._DSSA(negative_type, anchor_subject, anchor_activity, ii)
                elif ii % 9 == 6:
                    neg_sub, possible_clips = self._SSDA(negative_type, anchor_subject, anchor_activity, ii)
            elif ii % 3 == 1:
                neg_sub, possible_clips = self._SSSA(negative_type, anchor_subject, anchor_activity, ii)
            else:
                neg_sub, possible_clips = self._ADJ(negative_type, anchor_subject, anchor_activity, ii)

        return neg_sub, possible_clips


if __name__ == '__main__':
    pass