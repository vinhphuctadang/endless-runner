# To add a new cell, type '
# To add a new markdown cell, type'
#
# Vinh Phuc Ta Dang ft Dao Cong Tinh
#
import os
import cv2
import time
import posenet
import numpy as np
from constants import *
# import tensorflow as tf
from scipy.spatial.distance import euclidean
from tensorflow.compat.v1.keras import backend as K

VID_EXT_VALIDS = ['.mp4', '.mov']
scale_factor = 0.5
VIDEO_URI = 0
GUI_Enable = True
oo = 1e9



def normalize(keypoint_coords):
    first_keypoint = keypoint_coords[0]
    nose, feature = first_keypoint[0], []
    mn, mx = oo, -oo
    for i in range(1, len(first_keypoint)):
        dist = euclidean(nose, first_keypoint[i])
        feature.append(dist)

    feature = np.array(feature)
    mx = np.max(feature)
    mn = np.min(feature)
    # norm = np.linalg.norm(feature)
    # normalized_feature = feature/norm
    normalized_feature = np.array([(val-mn)/(mx-mn) for val in feature])
    return normalized_feature

def extractFeatures(VIDEO_URI):
    sess = K.get_session()
    model_cfg, model_outputs = posenet.load_model(101, sess)
    output_stride = model_cfg['output_stride']

    cap = cv2.VideoCapture(VIDEO_URI)
    flip = False
    cap.set(3, 257)
    cap.set(4, 257)
    frame_count = 0
    frame_series, frame_seq = [], []

    print('Extracting video ' + VIDEO_URI + '...')
    while True:
        try:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, flip=flip, scale_factor=scale_factor, output_stride=output_stride)
        except:
            print("End video %s.%d\n" % (VIDEO_URI, len(frame_series)))
            break

        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
            model_outputs,
            feed_dict={'image:0': input_image}
        )

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
            heatmaps_result.squeeze(axis=0),
            offsets_result.squeeze(axis=0),
            displacement_fwd_result.squeeze(axis=0),
            displacement_bwd_result.squeeze(axis=0),
            output_stride=output_stride,
            min_pose_score=0.15)

        keypoint_coords *= output_scale
        normalized_feature = normalize(keypoint_coords)
        frame_count += 1
        frame_seq.append(normalized_feature)
        if frame_count % window_size == 0:
            frame_series.append(frame_seq)
            frame_seq = []

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        if GUI_Enable:
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            cv2.imshow('posenet', overlay_image)
            # if "running" in VIDEO_URI:
            #     filename = "running.png"
            # else:
            #     filename = "walking.png"

            # filename = time.strftime('%Y%m%d_%H%M%S') + ".png"
            cv2.imwrite(filename, overlay_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return np.array(frame_series)


actions = np.array([])
labels = np.array([])
datasets_dir = 'pose/datasets'

time_start = time.time()
for action in classes:
    action_dir = os.path.join(datasets_dir, action)
    if '.DS_Store' not in action_dir:
        print("\n############# ACTION: %s #############\n" % action)
        total = 0
        for vid in os.listdir(action_dir):
            name, ext = os.path.splitext(vid)
            if ext in VID_EXT_VALIDS and name in "Tinh_Running":
                video_path = os.path.join(action_dir, vid)
                keyPoints = extractFeatures(video_path)
                labels = np.append(labels, np.array(
                    [action]*keyPoints.shape[0]))
                total += keyPoints.shape[0]
                if len(actions) == 0:
                    actions = keyPoints
                else:
                    actions = np.vstack((actions, keyPoints))
        print("Total: %d" % total)
time_end = time.time()


