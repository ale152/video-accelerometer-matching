import json
import os
import zipfile
from io import BytesIO
from math import floor

import cv2
import numpy as np
from PIL import Image
from imageio import imsave
from scipy.misc import imresize, imread
from tqdm import tqdm


if __name__ == '__main__':
    data_path = r'G:\SPHERE_Calorie\SPHERE_Calorie'
    out_path = r'better_silhouette'
    sub_template = 'Subject{}_Record{}'
    img_folder = 'rgb'
    dep_folder = 'depth'
    ske_folder = 'skeleton'
    img_template = 'img_%d.png'
    dep_template = 'depth_%d.png'
    ske_template = 'img_%d_keypoints.json'
    subjects = [1, 10]

    skel_conn = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                [1, 0], [0, 14], [14, 16], [0, 15]]

    for sub in range(subjects[0], subjects[1]+1):
        for rec in range(1, 3):
            # Define the paths
            sub_name = sub_template.format(sub, rec)
            sub_path = os.path.join(data_path, sub_name)
            img_path = os.path.join(sub_path, img_folder)
            dep_path = os.path.join(sub_path, dep_folder)
            ske_path = os.path.join(sub_path, ske_folder)
            sub_out = os.path.join(sub_path, out_path)
            if not os.path.exists(sub_out):
                os.makedirs(sub_out)

            # Read the bounding boxes
            bbox = np.loadtxt(os.path.join(sub_path, 'userBB0.txt')).astype('int') # [:, 9:13]
            frames = bbox[:, 0]

            for i in tqdm(range(len(frames))):
                # Open the image
                file_path = os.path.join(img_path, img_template % frames[i])
                try:
                    img = imread(file_path)
                except FileNotFoundError:
                    print('Impossible to read {}'.format(file_path))
                    continue

                # Open the depth map
                file_path = os.path.join(dep_path, dep_template % frames[i])
                try:
                    dep = imread(file_path)
                except FileNotFoundError:
                    print('Impossible to read {}'.format(file_path))
                    continue

                # Open the skeleton
                file_path = os.path.join(ske_path, ske_template % frames[i])
                with open(file_path) as file:
                    ske = json.load(file)

                # Transform the skeleton into a mask
                ske_mask = np.zeros(img.shape[:2], np.uint8)
                for person in ske['people']:
                    n_joints = len(person['pose_keypoints']) // 3
                    for joint in range(n_joints):
                        xj, yj = person['pose_keypoints'][joint*3 + 0], person['pose_keypoints'][joint*3 + 1]
                        xj, yj = int(xj), int(yj)
                        try:
                            ske_mask[yj, xj] = cv2.GC_FGD  # The skeleton joints are SURE foreground
                        except IndexError:
                            continue
                    # Print line
                    for conn in skel_conn:
                        x1, y1 = person['pose_keypoints'][conn[0]*3 + 0], person['pose_keypoints'][conn[0]*3 + 1]
                        x2, y2 = person['pose_keypoints'][conn[1]*3 + 0], person['pose_keypoints'][conn[1]*3 + 1]
                        x1, y1 = int(x1), int(y1)
                        x2, y2 = int(x2), int(y2)
                        if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:
                            cv2.line(ske_mask, (x1, y1), (x2, y2), cv2.GC_FGD)

                # Dilate the joints
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                # ske_mask = cv2.dilate(ske_mask, kernel, iterations=1)
                ske_mask[ske_mask == 0] = cv2.GC_PR_FGD  # The bounding box will be "PROBABLY foreground"
                # Make sure that everything outside the bounding box is set to zero, even though openpose detected
                # someone
                bbox_i = bbox[i, 9:13].astype('int')
                mask = np.zeros(img.shape[:2], np.uint8)
                mask[:] = cv2.GC_BGD  # Outside the bounding box is SURE background
                mask[bbox_i[3]:bbox_i[1], bbox_i[2]:bbox_i[0]] = ske_mask[bbox_i[3]:bbox_i[1], bbox_i[2]:bbox_i[0]]


                img_crop = img[bbox_i[3]:bbox_i[1], bbox_i[2]:bbox_i[0], :]
                dep_crop = dep[bbox_i[3]:bbox_i[1], bbox_i[2]:bbox_i[0]]

                # # Normalise the depth around the bounding box
                # dep = dep.astype('float')
                # min_dep = dep_crop[dep_crop > 0].min()
                # max_dep = dep_crop[dep_crop > 0].max()
                # dep_norm = (dep - min_dep) / (max_dep - min_dep)
                # dep_norm[dep_norm < 0] = 0
                # dep_norm[dep_norm > 1] = 1
                # dep_norm = 1 - dep_norm
                #
                # # Fill the holes
                # bad_dep = dep==0
                # filling = np.random.random(dep.shape)
                # dep_norm[bad_dep] = filling[bad_dep]

                dep_norm = ((dep.astype('float')- dep_crop[dep_crop>0].min())/dep_crop.max()*255).astype('uint8')

                # dep = (dep * 255).astype('uint8')
                dep_norm = np.dstack((dep_norm, dep_norm, dep_norm))
                # img = (img.astype('float') * dep_norm).astype('uint8')

                try:
                    bgdModel
                    fgdModel
                except NameError:
                    bgdModel = np.zeros((1, 65), np.float64)
                    fgdModel = np.zeros((1, 65), np.float64)

                rect = (bbox_i[2], bbox_i[3], bbox_i[0], bbox_i[1])
                try:
                    mask, bgdModel, fgdModel = cv2.grabCut(dep_norm, mask, rect, bgdModel, fgdModel, 5,
                                                         cv2.GC_INIT_WITH_MASK)
                except:
                    print('Something went wrong with subj {}'.format(os.path.join(img_path, img_template % frames[i])))

                silhouette = np.logical_or(mask == cv2.GC_PR_FGD, mask == cv2.GC_FGD).astype(np.uint8) * 255
                file_out = os.path.join(sub_out, 'sil_%d.png' % frames[i])
                imsave(file_out, silhouette)
