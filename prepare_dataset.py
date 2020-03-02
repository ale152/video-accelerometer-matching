import os
import cv2
import zipfile
from io import BytesIO
import numpy as np
from PIL import Image
from imageio import imread
from scipy.misc import imresize
from tqdm import tqdm
from numpy.lib.stride_tricks import as_strided


def resize_image(img, bbox, frame_size, keep_AR=True, expand_edges=False):
    cropbox = img[bbox[3]:bbox[1], bbox[2]:bbox[0]]

    if keep_AR:
        AR = cropbox.shape[0] / cropbox.shape[1]
        # Vertically elongated
        if AR > 1:
            # The height will match the target size
            new_size = [frame_size, np.round(frame_size / AR)]
        # Horizontally elongated
        else:
            # The width will match the target size
            new_size = [np.round(frame_size * AR), frame_size]

        new_size = np.round(new_size).astype('int')
        # If the image is smaller than the target size, don't resize it
        if np.all([bf < frame_size for bf in cropbox.shape]):
            new_size = cropbox.shape
        else:
            cropbox = imresize(cropbox, new_size, interp='nearest')

        if expand_edges:
            lef = np.round(frame_size / 2 - new_size[1] / 2).astype('int')
            rig = frame_size  - lef - new_size[1]
            top = np.round(frame_size / 2 - new_size[0] / 2).astype('int')
            bot = frame_size - top - new_size[0]
            imgbox = cv2.copyMakeBorder(cropbox, top, bot, lef, rig, cv2.BORDER_REPLICATE)
        else:
            first_row = np.round(frame_size / 2 - new_size[1] / 2).astype('int')
            first_col = np.round(frame_size / 2 - new_size[0] / 2).astype('int')
            imgbox = np.zeros((frame_size, frame_size, 3), dtype=img.dtype)
            imgbox[first_col:first_col + new_size[0], first_row:first_row + new_size[1], :] = cropbox

        return imgbox
    else:
        cropbox = imresize(cropbox, [frame_size, frame_size], interp='bilinear')
        return cropbox


def windowed_view(arr, window, overlap):
    """From https://stackoverflow.com/questions/18247009/window-overlap-in-pandas"""
    arr = np.asarray(arr)
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step, window)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) + arr.strides[-1:])
    return as_strided(arr, shape=new_shape, strides=new_strides)


def save_image_to_zip(img, zip, name):
    # Save each frame to file into the buffer
    buffer = BytesIO()
    image = Image.fromarray(img)
    image.save(buffer, format='png')
    buffer.seek(0)
    zip.writestr(name, buffer.getvalue())


def save_acc_to_zip(acc_data, acc_time, zip, name):
    buffer = BytesIO()
    data = np.hstack((acc_time, acc_data))
    np.savetxt(buffer, data, fmt='%s', delimiter=',', newline='\n')
    buffer.seek(0)
    zip.writestr(name, buffer.getvalue())


def save_boxes_to_zip(boxes, zip, name):
    buffer = BytesIO()
    np.savetxt(buffer, boxes, fmt='%s', delimiter=',', newline='\n')
    buffer.seek(0)
    zip.writestr(name, buffer.getvalue())


if __name__ == '__main__':
    data_path = r'G:\SPHERE_Calorie\SPHERE_Calorie'
    out_path = r'F:\SPHERE_Calorie\ReID\acc_sil_100_00_clean'
    sub_template = 'Subject{}_Record{}'
    img_folder = 'better_silhouette'
    img_template = 'sil_%d.png'
    img_time_file = 'frameTSinfo0.000000_all.txt'
    acc_folder1 = 'ACC0.000000'
    acc_folder2 = 'ACC1.000000'
    acc_file1 = 'ACC_0.000000.txt'
    acc_file2 = 'ACC_1.000000.txt'
    clip_length = 100
    keep_AR = True
    expand_edges = False  # True for rgb, False for silhouette
    # img_folder = 'silhouette'
    # img_template = 'sil_%d.png'
    # keep_AR = True
    frame_size = 100  # 224 for MobileNet
    frame_overlap = 0  # 0
    subjects = [1, 10]

    activities = ('none', 'standing', 'sitting', 'walking', 'wiping', 'vacuuming', 'sweeping', 'lying', 'exercising',
                  'stretching', 'cleaning', 'reading')

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for sub in tqdm(range(subjects[0], subjects[1]+1), 'Subject'):
        for rec in tqdm(range(1, 3), 'Session'):
            # Define the paths
            sub_name = sub_template.format(sub, rec)
            sub_path = os.path.join(data_path, sub_name)
            img_path = os.path.join(sub_path, img_folder)
            acc1_path = os.path.join(sub_path, acc_folder1, acc_file1)
            acc2_path = os.path.join(sub_path, acc_folder2, acc_file2)
            img_time_path = os.path.join(sub_path, img_time_file)

            # Create a zipfile per subject
            sub_name_zip = os.path.join(out_path, sub_name + '.zip')
            zip_sub = zipfile.ZipFile(sub_name_zip, 'w')

            # Read the labels file
            labels = np.loadtxt(os.path.join(sub_path, 'labels.txt'), delimiter=',').astype('int') # [:, 9:13]

            # Read the accelerometer file
            acc_data1 = np.loadtxt(acc1_path)
            acc_time1 = acc_data1[:, 2]
            acc_xyz1 = acc_data1[:, 3:6]
            acc_data2 = np.loadtxt(acc2_path)
            acc_time2 = acc_data2[:, 2]
            acc_xyz2 = acc_data2[:, 3:6]

            # Read the frames synchronisation file
            img_time = np.loadtxt(img_time_path)[:, 6]

            # Read the bounding boxes
            bbox = np.loadtxt(os.path.join(sub_path, 'userBB0.txt')).astype('int') # [:, 9:13]
            frames = bbox[:, 0]

            # Read the activities per each frame
            labels_indexed = np.zeros((np.maximum(labels[:, 0].max(), frames.max()) + 1))
            for lab in labels:
                labels_indexed[lab[0]] = lab[1]

            # Loop over all the activities first
            for act_i in tqdm(range(len(activities)), 'Processing activities', position=0):
                # Find frames performing this activity
                act_frames = frames[labels_indexed[frames] == act_i]
                if len(act_frames) < clip_length:
                    print('{} has no activity {}'.format(sub_name, activities[act_i]))
                    continue

                windowed_frames = windowed_view(act_frames, clip_length, frame_overlap)

                # Divide it into clips
                n_clips = windowed_frames.shape[0]
                for clip_i in tqdm(range(n_clips), 'Processing clips', position=0):
                    # Load the accelerations, the clips and the bounding boxes first, and then save them all to the
                    # zip. This is done to avoid having clips containing only acceleration or images.
                    clip_name = 'clip_%05d' % clip_i
                    clip_frame_ids = windowed_frames[clip_i, :]

                    # Synchronise with the accelerometers
                    t_start = img_time[clip_frame_ids[0]]
                    t_end = img_time[clip_frame_ids[-1]]

                    if (t_end - t_start)/100 > 35:
                        print('(!) Skipping clip {} because initial and final frames are too far apart'.format(clip_name))
                        continue

                    acc_sync1 = acc_xyz1[np.logical_and(acc_time1 >= t_start, acc_time1 <= t_end), :]
                    acc_time_sync1 = acc_time1[np.logical_and(acc_time1 >= t_start, acc_time1 <= t_end)][:, None]
                    arcname1 = os.path.join(acc_folder1, activities[act_i], clip_name, 'acc.csv')

                    acc_sync2 = acc_xyz2[np.logical_and(acc_time2 >= t_start, acc_time2 <= t_end), :]
                    acc_time_sync2 = acc_time2[np.logical_and(acc_time2 >= t_start, acc_time2 <= t_end)][:, None]
                    arcname2 = os.path.join(acc_folder2, activities[act_i], clip_name, 'acc.csv')

                    if acc_sync1.size == 0 or acc_sync2.size == 0:
                        print('Skipping clip {} because no acceleration was found'.format(clip_name))
                        continue

                    clip_boxes = np.zeros((clip_length, 12))
                    clip_frames = []
                    clip_frame_names = []
                    # Load the frames for this clip
                    for i in range(clip_length):
                        # Open the image and resize it
                        file_path = os.path.join(img_path, img_template % clip_frame_ids[i])
                        try:
                            img = imread(file_path)
                            if len(img.shape) < 3:
                                img = np.dstack((img, img, img))
                        except Exception as error:
                            print('Error while processing {}. {}'.format(file_path, error))
                            break
                        try:
                            box_frame = np.where(bbox[:, 0] == clip_frame_ids[i])[0][0]
                            img = resize_image(img, bbox[box_frame, 9:13],
                                               frame_size, keep_AR=keep_AR, expand_edges=expand_edges)
                        except IndexError as err:
                            print('Impossible to resize {}. {}'.format(file_path, err))
                            break

                        # Save it into the archive
                        arcname = os.path.join(img_folder, activities[act_i], clip_name, img_template % i)
                        clip_frame_names.append(arcname)
                        clip_frames.append(img)

                        # Store the bounding box for these frames
                        clip_boxes[i, ...] = bbox[box_frame, 1:]

                    if len(clip_frames) < clip_length:
                        continue

                    # Save the frames
                    for i in range(clip_length):
                        save_image_to_zip(clip_frames[i], zip_sub, clip_frame_names[i])

                    # Save the bounding boxes
                    arcname = os.path.join('bounding_boxes', activities[act_i], clip_name, 'boxes.csv')
                    save_boxes_to_zip(clip_boxes, zip_sub, arcname)

                    # Save the acceleration
                    save_acc_to_zip(acc_sync1, acc_time_sync1, zip_sub, arcname1)
                    save_acc_to_zip(acc_sync2, acc_time_sync2, zip_sub, arcname2)

            zip_sub.close()
