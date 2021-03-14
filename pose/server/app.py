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
# from constants import *
from scipy.spatial.distance import euclidean
from tensorflow.keras.models import load_model
from tensorflow.compat.v1.keras import backend as K

from threading import Thread, Lock
from flask import Flask

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

scale_factor = 0.5
GUI_Enable = True
VIDEO_URI = 0
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale, thickness = 0.75, 2

app             = Flask(__name__)
device          = None
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
        for coord in keypoint_coords[0]: # only consider pose 0
            tmp_coords.append(
                {
                    'x': coord[0],
                    'y': coord[1],
                    'c': keypoint_scores[0][i]
                }
            )
            i += 1
        result = {
            "code": 1, 
            "result" : tmp_coords,
        }
    else:
        result = {
            "code": -1,
            "message": "No pose detected"
        }, 500
    mutex.release()
    return result

# @app.route("/init")
# def init():
#     try:
        
#     except Exception as e:
#         return {"code": -1, "message": str(e)}, 500
#     return {"code": 1}

# @app.route("/done")
# def done():
#     try:
#         cap.release()
#         sess.close()
#     except Exception as e:
#         return {"code": -1, "message": str(e)}, 500
#     return {"code": 1}

def do_workflow():
    global frame_count, frame_series, frame_seq, label, model_cfg, model_outputs, output_stride
    global pose_scores, keypoint_scores, keypoint_coords

    sess = K.get_session()
    model_cfg, model_outputs = posenet.load_model(101, sess)
    output_stride = model_cfg['output_stride']

    cap = cv2.VideoCapture(VIDEO_URI)
    flip = True
    cap.set(3, 257)
    cap.set(4, 257)
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
    cap.release()
if __name__=="__main__":
    main_thread = Thread(target=do_workflow, args=())
    main_thread.start()
    app.run(debug=True)

    