# # # # # # # # # # # # # # # #
# Vinh Phuc Ta Dang ft Dao Cong Tinh
# # # # # # # # # # # # # # # #
import os
import sys
# add search path
sys.path.append("../../")

import cv2
import pickle
import posenet
import numpy as np
from threading import Thread, Lock
from scipy.spatial.distance import euclidean
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response
from sklearn.metrics import pairwise_distances as distance
from tensorflow.compat.v1.keras.backend import get_session


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

scale_factor = 0.4
window_size = 10
VIDEO_URI = 0
SEQUENCE_MODEL_DIR = "/Users/dcongtinh/Workspace/endless-runner/results/20210530_154012_LSTM_Action_Tanh/k6/LSTM_Action_Tanh.h5"
STAND_CRUNCH_MODEL_DIR = "_models/stand_crunch.model"
SCORE_THRESHOLD = 0.15

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale, thickness = 1.5, 2

app = Flask(__name__)

cap             = None
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

mutex = Lock()


@app.route("/")
def ping():
    if not cap:
        # still not ready
        return {"code": 0}
    # ready
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

def b64encode(image):
    '''
    image: array data
    '''
    import base64
    success, img_enc = cv2.imencode('.JPEG', image)
    data = base64.b64encode(img_enc)
    img = data.decode('ascii')
    img = 'data:image/jpeg;base64,%s' % img
    return img


@app.route("/show_image")
def show_image():
    return "<img style='display: flex; margin: auto' src='%s'/>" % b64encode(get_img)


@app.route("/image")
def get_image():
    import base64
    success, img_enc = cv2.imencode('.JPEG', get_img)
    data = base64.b64encode(img_enc)
    img = data.decode('ascii')
    return img


@app.route("/video")
def get_images():
    return render_template('video.html')


def gen():
    while True:
        success, img_enc = cv2.imencode('.JPEG', get_img)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'%s' % (img_enc.tobytes()) + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def normalize(features):
    mx = np.max(features)
    mn = np.min(features)
    # print(mx, mn)
    for feature in features:
        for i in range(len(feature)):
            feature[i] = (feature[i] - mn) / (mx - mn)

    return np.array(features)


def extract_feature_sequence(frame_seq):
    features = []
    kp_prev = frame_seq[0] # keypoint prev
    for i in range(1, len(frame_seq)):
        kp_curr = frame_seq[i]
        dist = []
        for p_i, p_j in zip(kp_curr, kp_prev):
            dist.append(euclidean(p_i, p_j))
        features.append(dist)

    return normalize(features)


def extract_feature_stand_scruch(keypoint_coords):
    features = distance(keypoint_coords[0:1], keypoint_coords[1:])[0]
    # normalize
    mx = max(features)
    mn = min(features)
    if mx == 0:
        return np.array([0]*len(keypoint_coords[1:]))

    for index in range(len(features)):
        features[index] = (features[index] - mn) / (mx-mn)
    return features


def predict_stand_crunch(model, keypoint_coords):
    feature = extract_feature_stand_scruch(keypoint_coords)
    # print("posture feature:", feature)

    # detect number of zero feature:
    known_points_count = len(feature[feature > 0.])
    # if too many zero then return unknown
    if known_points_count < MIN_KEYPOINT_TO_PREDICT:
        return "unknown", None
    # otherwise predict
    try:
        pred = model.predict_proba([feature])[0]
        idx = np.argmax(pred)
        posture_label = STAND_CRUNCH_LABELS[idx]
    except Exception as e:
        # if error happend returns unknown
        print("Error happened:", e)
        posture_label = "unknown"
    # return label
    return posture_label, pred[idx]


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

# # This route should be implemented in release mode
# @app.route("/done")
# def done():
#     try:
#         cap.release()
#         sess.close()
#     except Exception as e:
#         return {"code": -1, "message": str(e)}, 500
#     return {"code": 1}


def do_workflow():
    global cap, frame_seq, label, model_cfg, model_outputs, output_stride
    global pose_scores, keypoint_scores, keypoint_coords

    global actions, model_loaded, current_action, stand_crunch_model, current_stand_crunch_status, current_lane
    global get_img
    #
    # load model:
    #
    model_loaded = load_model(SEQUENCE_MODEL_DIR)

    # load stand and crunch model
    with open(STAND_CRUNCH_MODEL_DIR, "rb") as f:
        stand_crunch_model = pickle.load(f)
    # load sessions and related DL components

    sess = get_session()
    model_cfg, model_outputs = posenet.load_model(101, sess)
    output_stride = model_cfg['output_stride']

    cap = cv2.VideoCapture(VIDEO_URI)
    flip = True
    frame_seq = []
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
            min_pose_score=SCORE_THRESHOLD)
        keypoint_coords *= output_scale
        mutex.release()
        first_keypoint = keypoint_coords[0]
        # predict lane and stand_crunch status
        current_stand_crunch_status, proba = predict_stand_crunch(stand_crunch_model, first_keypoint)
        current_lane = get_lane(display_image, first_keypoint)

        if current_stand_crunch_status == "stand":
            frame_seq.append(first_keypoint)
            if len(frame_seq) % window_size == 0:
                normalized_feature = extract_feature_sequence(frame_seq)
                normalized_feature = np.expand_dims(normalized_feature, axis=0)
                pred = model_loaded.predict(normalized_feature)[0]
                idx = np.argmax(pred)
                mutex.acquire()
                current_action = actions[idx]
                label = "%s - %.2f; %s" % (current_action, pred[idx], current_lane)
                mutex.release()
                frame_seq = []
        elif current_stand_crunch_status == "crunch":
            label = "crunch - %.2f; %s" % (proba, current_lane)


        # render for demo purpose
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=SCORE_THRESHOLD, min_part_score=SCORE_THRESHOLD)

        cv2.putText(overlay_image, label, (20, 40),
                    fontFace, fontScale=fontScale, color=(0, 0, 255), thickness=thickness)
        get_img = overlay_image

    cap.release()


if __name__ == "__main__":
    main_thread = Thread(target=do_workflow, args=())
    main_thread.start()
    app.run(host="0.0.0.0", debug=True)
