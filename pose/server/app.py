
# # # # # # # # # # # # # # # #
# Vinh Phuc Ta Dang ft Dao Cong Tinh
# # # # # # # # # # # # # # # #
import os
import sys
import pickle
# add search path
sys.path.append("../../")

import cv2
import posenet
import numpy as np
from constants import *
from threading import Thread, Lock

from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances as distance
from tensorflow.keras.models import load_model
from flask import Flask, Response
from tensorflow.compat.v1.keras.backend import get_session

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

scale_factor = 0.5
VIDEO_URI = 0
MODEL_DIR = "_models/pose_clf_20210324_090942.h5"
STAND_CRUNCH_MODEL_DIR = "_models/stand_crunch.model"
LABEL_DIR = "_models/labels.npy"

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale, thickness = 0.75, 2

app = Flask(__name__)

cap             = None
frame_series    = None
frame_seq       = None

# tf session
sess            = None
model_cfg       = None
model_outputs   = None
output_stride   = None

pose_scores     = None
keypoint_scores = None
keypoint_coords = None

get_img         = None

# model and action
current_action  = None
model_loaded    = None
stand_crunch_model = None 

current_stand_crunch_status = None 
current_lane = None

# constants
actions = ["idle", "running", "walking"]
STAND_CRUNCH_LABELS = ["stand", "crunch"]
LANE_LABELS = ["left", "middle", "right"]
MIN_KEYPOINT_TO_PREDICT = 8


# mutex for getting rid of race condition
mutex = Lock()

def extract_feature(keypoint_coords):
    features = distance(keypoint_coords[0:1], keypoint_coords[1:])[0]
    # normalize
    mx = max(features)
    mn = min(features)
    for index in range(len(features)):
        features[index] = (features[index] - mn) / (mx-mn)
    return features

def predict_stand_crunch(model, keypoint_coords):
    feature = extract_feature(keypoint_coords)
    print("posture feature:", feature)

    # detect number of zero feature:
    known_points_count = len(feature[feature > 0.])
    # if too many zero then return unknown
    if known_points_count < MIN_KEYPOINT_TO_PREDICT:
        return "unknown"
    # otherwise predict
    try:
        posture = model.predict([feature])[0]
        posture_label = STAND_CRUNCH_LABELS[posture]
    except Exception as e:
        # if error happend returns unknown
        print("Error happened:", e)
        posture_label = "unknown"
    # return label
    return posture_label

# return lane in within frame
def get_lane(frame, keypoint_coords):
    
    x1 = frame.shape[1] // 3
    x2 = frame.shape[1] // 3 * 2

    # for simplicity, just base on nose position
    nose = keypoint_coords[0]

    # nose[1] presents nose on x-axis
    if nose[1] <= x1: 
        return LANE_LABELS[0]
    if nose[1] <= x2:
        return LANE_LABELS[1]

    return LANE_LABELS[2]

@app.route("/")
def ping():
    return {"code": 1}

@app.route("/pose")
def get_pose():

    return {
        "code": 1, 
        "action": current_action, 
        "stand_crunch": current_stand_crunch_status, 
        "lane": current_lane
    }

@app.route("/init")
def init():
    try:
        device = cv2.VideoCapture(VIDEO_URI)
    except Exception as e:
        return {"code": -1, "message": str(e)}
    return {"code": 1}

def gen():
    while True:
        success, img_enc = cv2.imencode('.JPEG', get_img)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'%s' % (img_enc.tobytes()) + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route("/done")
# def done():
#     try:
#         cap.release()
#         sess.close()
#     except Exception as e:
#         return {"code": -1, "message": str(e)}, 500
#     return {"code": 1}


def do_workflow():
    global cap, frame_series, frame_seq, label, model_cfg, model_outputs, output_stride
    global pose_scores, keypoint_scores, keypoint_coords

    global actions, model_loaded, current_action, stand_crunch_model, current_stand_crunch_status, current_lane
    global get_img
    #
    # load model:
    #
    model_loaded = load_model(MODEL_DIR)

    # load stand and crunch model
    with open(STAND_CRUNCH_MODEL_DIR, "rb") as f:
        stand_crunch_model = pickle.load(f)
    # load sessions and related DL components

    sess = get_session()
    model_cfg, model_outputs = posenet.load_model(101, sess)
    output_stride = model_cfg['output_stride']

    cap = cv2.VideoCapture(VIDEO_URI)
    flip = True
    # cap.set(3, 257)
    # cap.set(4, 257)
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
        mutex.acquire()
        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
            heatmaps_result.squeeze(axis=0),
            offsets_result.squeeze(axis=0),
            displacement_fwd_result.squeeze(axis=0),
            displacement_bwd_result.squeeze(axis=0),
            output_stride=output_stride,
            min_pose_score=0.25)
        keypoint_coords *= output_scale
        mutex.release()

        first_keypoint = keypoint_coords[0]
        if len(np.unique(first_keypoint)) > 1:
            nose = first_keypoint[0]
            feature = []
            for i in range(1, len(first_keypoint)):
                feature.append(euclidean(nose, first_keypoint[i]))
            frame_seq.append(feature)
            if len(frame_seq) % window_size == 0:
                frame_seq = np.array(frame_seq)
                frame_seq = np.expand_dims(frame_seq, axis=0)
                pred = model_loaded.predict(frame_seq)[0]
                pred = np.argmax(pred)
                mutex.acquire()
                current_action = actions[pred]
                label = current_action
                mutex.release()
                frame_seq = []
        else:
            label = "idle"
            frame_seq = []


        # predict lane and stand_crunch status
        current_stand_crunch_status = predict_stand_crunch(stand_crunch_model, first_keypoint)
        current_lane = get_lane(display_image, first_keypoint)
        
        # render for demo purpose
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.15)

        cv2.putText(overlay_image, label, (20, 20),
                    fontFace, fontScale=fontScale, color=(0, 255, 0), thickness=thickness)
        get_img = overlay_image

    cap.release()
    # cv2.destroyAllWindows()


if __name__=="__main__":
    # start new thread in order to allow camera to work
    Thread(target=do_workflow,).start()

    # start server
    app.run(debug=True)
    