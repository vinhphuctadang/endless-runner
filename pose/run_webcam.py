#
# Vinh Phuc Ta Dang ft Dao Cong Tinh
#
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import math
import cv2

MODEL_PATH = 'model/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
BODY_SIZE = (257, 257)
DST_SIZE = (257, 257)
THRESHOLD = 0.9


ratio = BODY_SIZE[0] / DST_SIZE[0]
VIDEO_URI = 0 # '/Users/dcongtinh/Workspace/endless-runner/pose/datasets/running/Tinh_Running.mov'

# VIDEO_URI = '/Users/dcongtinh/Workspace/endless-runner/pose/walking.mov'
# # parts of pose
# # below constants referred from https://github.com/rwightman/posenet-python
# # special thanks for the repo authors
PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]

CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
]


actions = np.load('labels.npy')
actions = np.unique(actions)
model_loaded = load_model('../research/lstm_keras.h5')
# Doing the legend steps:
# tf.lite.Interpreter() -> transform data -> run inferences -> Interprete output

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale, thickness = 0.75, 2


def sigmoid(x):
    return 1/(1 + math.exp(-x))


def main():

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    print('Starting camera...')
    # input_data = cv2.imread('./pose_scaled.png')

    cap = cv2.VideoCapture(VIDEO_URI)
    num_frame, frame_seq = 0, []
    label = 'idle'

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End Video.")
            break
        num_frame += 1
        # pivotY = (frame.shape[0] - BODY_SIZE[0]) // 2
        # pivotX = (frame.shape[1] - BODY_SIZE[0]) // 2
        # frame = cv2.flip(frame[pivotY:pivotY+BODY_SIZE[1],
        #                        pivotX:pivotX+BODY_SIZE[0]], 1)
        if type(VIDEO_URI) == int:
            frame = cv2.flip(frame, 1)
        else:
            frame = frame[:, ::-1]
        #--------------------------------------------------------------------------#
        frame = cv2.resize(frame, DST_SIZE)  # scale image
        frame = frame.astype(np.float32) / 255.0
        interpreter.set_tensor(input_details[0]['index'], [frame])
        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        heatmaps = interpreter.get_tensor(output_details[0]['index'])
        offsets = interpreter.get_tensor(output_details[1]['index'])

        # offsets = interpreter.graph.get_tensor('offset:0')
        # displacement_fwd = interpreter.get_tensor_by_name('displacement_fwd_2:0')
        # displacement_bwd = interpreter.get_tensor_by_name('displacement_bwd_2:0')
        # heatmaps = interpreter.get_tensor_by_name('heatmap:0')

        #--------------------------------------------------------------------------#
        # code inferred from https://github.com/tensorflow/examples/blob/master/lite/examples/posenet/android/posenet/src/main/java/org/tensorflow/lite/examples/posenet/lib/Posenet.kt
        imgHeight = frame.shape[0]
        imgWidth = frame.shape[1]

        height, width, numKeypoints = heatmaps.shape[1], heatmaps.shape[2], heatmaps.shape[3]
        # Finds the (row, col) locations of where the keypoints are most likely to be.
        keypointPositions = [None] * numKeypoints
        for keypoint in range(numKeypoints):
            maxVal = heatmaps[0][0][0][keypoint]
            maxRow = 0
            maxCol = 0
            for row in range(height):
                for col in range(width):
                    if heatmaps[0][row][col][keypoint] > maxVal:
                        maxVal = heatmaps[0][row][col][keypoint]
                        maxRow = row
                        maxCol = col
            keypointPositions[keypoint] = (maxRow, maxCol)

        # # Calculating the x and y coordinates of the keypoints with offset adjustment.
        xCoords = [0] * numKeypoints
        yCoords = [0] * numKeypoints
        xyCoords = []
        confidenceScores = [0] * numKeypoints
        for idx in range(len(keypointPositions)):
            positionY = keypointPositions[idx][0]
            positionX = keypointPositions[idx][1]
            # compute coordination and confidence value
            yCoords[idx] = int(positionY / (height - 1) *
                               imgHeight + offsets[0][positionY][positionX][idx]) * int(ratio)
            xCoords[idx] = int(positionX / (width - 1) * imgWidth +
                               offsets[0][positionY][positionX][idx + numKeypoints]) * int(ratio)
            confidenceScores[idx] = sigmoid(
                heatmaps[0][positionY][positionX][idx])
            xyCoords.append(xCoords[idx])
            xyCoords.append(yCoords[idx])

        result = {}
        for idx in range(len(keypointPositions)):
            result[PART_NAMES[idx]] = {
                'x': xCoords[idx],
                'y': yCoords[idx],
                'c': confidenceScores[idx]
            }

        # print('Result:', json.dumps(result, indent=2))
        lined_frame = frame.copy()
        for key in result:
            if result[key]['c'] < THRESHOLD:
                continue
            point = result[key]['x'], result[key]['y']
            cv2.circle(frame, point, 1, (0, 255, 0), 2)
            cv2.circle(lined_frame, point, 1, (0, 255, 0), 2)
        
        # render result
        for edge in CONNECTED_PART_NAMES:
            if result[edge[0]]['c'] < THRESHOLD or result[edge[1]]['c'] < THRESHOLD: 
                continue
            cv2.line(
                lined_frame,
                (result[edge[0]]['x'], result[edge[0]]['y']),
                (result[edge[1]]['x'], result[edge[1]]['y']),
                (0, 0, 255),
                2
            )
        frame_seq.append(xyCoords)
        if num_frame % 30 == 0:
            # do something
            frame_seq = np.array(frame_seq)
            frame_seq = np.expand_dims(frame_seq, axis=0)
            pred = model_loaded.predict(frame_seq)
            pred = actions[np.argmax(pred)]
            label = pred
            frame_seq = []

        cv2.putText(frame, label, (20, 20),
                    fontFace, fontScale=fontScale, color=(0, 255, 0), thickness=thickness)

        cv2.imshow('frame', frame)
        cv2.imshow('lined_frame', lined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # destruction
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
