#
# Vinh Phuc Ta Dang ft Dao Cong Tinh
#
import os
import cv2
import math
import numpy as np
import tensorflow as tf

MODEL_PATH = 'model/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
BODY_SIZE = (257, 257)
DST_SIZE = (257, 257)
VID_EXT_VALIDS = ['.mp4', '.mov']

ratio = BODY_SIZE[0] / DST_SIZE[0]
n_fps = 30

VIDEO_URI = 0


interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def sigmoid(x):
    return 1/(1 + math.exp(-x))


def checkVideoValid(filename):
    for vid_ext in VID_EXT_VALIDS:
        if (vid_ext in filename):
            return True
    return False


def extractFeatures(VIDEO_URI):
    # Test the model on random input data.
    print('Extracting video ' + VIDEO_URI + '...')

    cap = cv2.VideoCapture(VIDEO_URI)
    num_frame = 0
    frame_series, frame_seq = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End video %s.%d\n" % (VIDEO_URI, len(frame_series)))
            break
        num_frame += 1

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

        frame_seq.append(xyCoords)
        if num_frame % 30 == 0:
            frame_series.append(frame_seq)
            frame_seq = []

    return np.array(frame_series)


def main():
    actions = np.array([])
    labels = np.array([])

    for action in os.listdir('datasets'):
        action_dir = os.path.join('datasets', action)
        if '.DS_Store' not in action_dir:
            print("\n############# ACTION: %s #############\n" % action)
            total = 0
            for vid in os.listdir(action_dir):
                _, ext = os.path.splitext(vid)
                if ext in VID_EXT_VALIDS:
                    video_path = os.path.join(action_dir, vid)
                    keyPoints = extractFeatures(video_path)
                    labels = np.append(labels, np.array(
                        [action]*keyPoints.shape[0]))
                    total += keyPoints.shape[0]
                    if len(actions) == 0:
                        actions = keyPoints
                    else:
                        actions = np.vstack((actions, keyPoints))
            print("Total: %d\n" % total)

    print('actions.shape =', actions.shape)
    print('labels.shape =', labels.shape)
    # np.save('actions.npy', actions)
    # np.save('labels.npy', labels)


if __name__ == '__main__':
    main()
