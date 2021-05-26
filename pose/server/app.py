# # # # # # # # # # # # # # # #
# Vinh Phuc Ta Dang ft Dao Cong Tinh
# # # # # # # # # # # # # # # #
import os
import sys
# add search path
sys.path.append("../../")

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

scale_factor = 0.4
VIDEO_URI = 0
MODEL_DIR = "_models/20210525_151927model.h5"
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
actions = ["idle", "running", "walking"]
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
    global pose_scores, keypoint_scores, keypoint_coords

    mutex.acquire()
    tmp_coords = []
    i = 0
    if keypoint_coords is not None:
        for coord in keypoint_coords[0]:  # only consider pose 0
            tmp_coords.append(
                {
                    'x': coord[0],
                    'y': coord[1],
                    'c': keypoint_scores[0][i]
                }
            )
            i += 1

        result = {
            "code": "SUCCESS",
            # "coords" : tmp_coords,
            "action": current_action,
        }
    else:
        result = {
            "code": "NO_POSE_DETECTED",
            # "coords": [],
            "action": "",
        }, 500
    mutex.release()
    return result


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

def extractFeature(frame_seq):
    features = []
    kp_prev = frame_seq[0] # keypoint prev
    for i in range(1, len(frame_seq)):
        kp_curr = frame_seq[i]
        dist = []
        for p_i, p_j in zip(kp_curr, kp_prev):
            dist.append(euclidean(p_i, p_j))
        features.append(dist)

    return normalize(features)


def do_workflow():
    global cap, frame_seq, label, model_cfg, model_outputs, output_stride
    global pose_scores, keypoint_scores, keypoint_coords

    global actions, model_loaded, current_action
    global get_img
    #
    # load model:
    #
    model_loaded = load_model(MODEL_DIR)

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

        frame_seq.append(keypoint_coords[0])
        if len(frame_seq) % window_size == 0:
            normalized_feature = extractFeature(frame_seq)
            normalized_feature = np.expand_dims(normalized_feature, axis=0)
            pred = model_loaded.predict(normalized_feature)[0]
            idx = np.argmax(pred)
            mutex.acquire()
            current_action = actions[idx]
            label = "%s - %.2f" % (current_action, pred[idx])
            mutex.release()
            frame_seq = []

        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=SCORE_THRESHOLD, min_part_score=SCORE_THRESHOLD)

        cv2.putText(overlay_image, label, (20, 40),
                    fontFace, fontScale=fontScale, color=(0, 255, 0), thickness=thickness)
        get_img = overlay_image

    cap.release()


if __name__ == "__main__":
    main_thread = Thread(target=do_workflow, args=())
    main_thread.start()
    app.run(host="0.0.0.0", debug=True)
