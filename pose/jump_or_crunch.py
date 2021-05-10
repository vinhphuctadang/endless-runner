#
# Vinh Phuc Ta Dang ft Dao Cong Tinh
#
import os
import sys
# add search path
sys.path.append("../")

import cv2
import posenet
import numpy as np
from constants import *
from threading import Thread, Lock
from scipy.spatial.distance import euclidean
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response
from tensorflow.compat.v1.keras.backend import get_session

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

scale_factor = 0.5
VIDEO_URI = "../data/stand.mov"
MODEL_DIR = "server/_models/pose_clf_20210324_090942.h5"
LABEL_DIR = "server/_models/labels.npy"

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale, thickness = 0.75, 2

app = Flask(__name__)

cap             = None
frame_series    = None
frame_seq       = None
frame_count     = None
# tf session
sess            = None
model_cfg       = None
model_outputs   = None
output_stride   = None

pose_scores     = None
keypoint_scores = None
keypoint_coords = None

get_img         = None

def main():
    model_loaded = load_model(MODEL_DIR)

    sess = get_session()
    model_cfg, model_outputs = posenet.load_model(101, sess)
    output_stride = model_cfg['output_stride']
    cap = cv2.VideoCapture(VIDEO_URI)
    flip = True

    # load model:
    #
    model_loaded = load_model(MODEL_DIR)

    sess = get_session()
    model_cfg, model_outputs = posenet.load_model(101, sess)
    output_stride = model_cfg['output_stride']
    cap = cv2.VideoCapture(VIDEO_URI)

    flip = True
    # cap.set(3, 257)
    # cap.set(4, 257)
    frame_count = 0
    frame_series, frame_seq = [], []
    label = 'idle'
    while True:
        input_image, display_image, output_scale = posenet.read_cap(
            cap, flip=flip, scale_factor=scale_factor, output_stride=output_stride
        )
        # without action prediction
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
            min_pose_score=0.25)
        keypoint_coords *= output_scale

        first_keypoint = keypoint_coords[0]
        nose = first_keypoint[0]
        for i in range(1, len(first_keypoint)):
            feature.append(euclidean(nose, first_keypoint[i]))

        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.15)
        cv2.putText(overlay_image, label, (20, 20), fontFace, fontScale=fontScale, color=(0, 255, 0), thickness=thickness)

        get_img = overlay_image
        cv2.imshow('Posenet', overlay_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    main()
