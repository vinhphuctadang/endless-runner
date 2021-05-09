#
# Vinh Phuc Ta Dang ft Dao Cong Tinh
#
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import math
import cv2
import pose_decoder
from constants import *

MODEL_PATH = 'model/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
BODY_SIZE = (257, 257)
DST_SIZE = (257, 257)
THRESHOLD = 0.9
ratio = BODY_SIZE[0] / DST_SIZE[0]

# Change URI to 0, i.e VIDEO_URI=0, to access camera
VIDEO_URI = "data/stand.mov" # "datatest/Phuc_Test.mp4"
# VIDEO_URI = '/Users/dcongtinh/Workspace/endless-runner/pose/walking.mov'
actions = np.load('labels.npy')
actions = np.unique(actions)
model_loaded = load_model('../research/lstm_keras.h5')
# Doing the legend steps:
# tf.lite.Interpreter() -> transform data -> run inferences -> Interprete output
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale, thickness = 0.75, 2

input_details = None 
output_details = None 
def get_pose(interpreter, frame):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], [frame])
    # execute the network
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    heatmaps = interpreter.get_tensor(output_details[0]['index'])
    offsets  = interpreter.get_tensor(output_details[1]['index'])
    displacement_fwd = interpreter.get_tensor(output_details[2]['index'])
    displacement_bwd = interpreter.get_tensor(output_details[3]['index'])
    # print("Heat shape:", heatmaps.squeeze(axis=0).shape)
    pose_scores, keypoint_scores, keypoint_coords = pose_decoder.decode_multiple_poses(
        heatmaps.squeeze(axis=0),
        offsets.squeeze(axis=0),
        displacement_fwd.squeeze(axis=0),
        displacement_bwd.squeeze(axis=0),
        output_stride=32, # (height - 1) / output_stride + 1 = 9, height = 257
        max_pose_detections=1,
        min_pose_score=0.1
    )

    keypoint_coord = keypoint_coords[0]
    keypoint_score = keypoint_scores[0]
    result = {}
    for idx in range(NUM_KEYPOINTS):
        result[PART_NAMES[idx]] = {
            'x': int(keypoint_coord[idx][1]),
            'y': int(keypoint_coord[idx][0]),
            'c': keypoint_score[idx]
        }
    return result

def render_pose(frame, result):
    for key in result:
        if result[key]['c'] == 0: 
            continue
        point = result[key]['x'], result[key]['y']
        # print("Point:", point)
        cv2.circle(frame, point, 1, (0, 255, 0), 2)
    # render result
    for edge in CONNECTED_PART_NAMES:
        if result[edge[0]]['c'] == 0 or result[edge[1]]['c'] == 0: 
            continue
        cv2.line(
            frame,
            (result[edge[0]]['x'], result[edge[0]]['y']),
            (result[edge[1]]['x'], result[edge[1]]['y']),
            (0, 0, 255),
            2
        )
    return frame

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def main():
    global input_details, output_details

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    print('Starting movie stream...')

    cap = cv2.VideoCapture(VIDEO_URI)
    num_frame, frame_seq = 0, []
    label = 'idle'

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End Video.")
            break
        num_frame += 1
        if type(VIDEO_URI) == int:
            frame = cv2.flip(frame, 1)
        # else:
        #     frame = frame[:, ::-1]
        #--------------------------------------------------------------------------#

        # precompute scale size
        scale_y = DST_SIZE[0]/frame.shape[0]
        scale_x = DST_SIZE[1]/frame.shape[1]

        # # scale frame to predict pose        
        predict_frame = cv2.resize(frame, BODY_SIZE)
        predict_frame = predict_frame.astype(np.float32) / 255.0
        pose = get_pose(interpreter, predict_frame)
        
        # # re-scale pose to fit original frame
        for key in pose:
            pose[key]["x"] = int(pose[key]["x"] / scale_x)
            pose[key]["y"] = int(pose[key]["y"] / scale_y)

        # # render and show frame
        render_pose(frame, pose)
        # nose = pose["nose"]

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # destruction
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
