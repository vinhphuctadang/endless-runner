#
# Vinh Phuc Ta Dang
#
import tensorflow as tf
import cv2
import numpy as np
import json
import math

MODEL_PATH = 'model/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
DST_SIZE = (257, 257)

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

# Doing the legend steps:
# tf.lite.Interpreter() -> transform data -> run inferences -> Interprete output


def sigmoid(x):
    return 1/(1 + math.exp(-x))


def main():

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    print('Loading image ...')
    input_data = cv2.imread('./acm-ptit.jpg')
    input_data = cv2.resize(input_data, DST_SIZE)  # scale image

    input_data = input_data.astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], [input_data])
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    heatmaps = interpreter.get_tensor(output_details[0]['index'])
    offsets = interpreter.get_tensor(output_details[1]['index'])

    # offsets = interpreter.graph.get_tensor('offset:0')
    # displacement_fwd = interpreter.get_tensor_by_name('displacement_fwd_2:0')
    # displacement_bwd = interpreter.get_tensor_by_name('displacement_bwd_2:0')
    # heatmaps = interpreter.get_tensor_by_name('heatmap:0')

    # code inferred from https://github.com/tensorflow/examples/blob/master/lite/examples/posenet/android/posenet/src/main/java/org/tensorflow/lite/examples/posenet/lib/Posenet.kt
    imgHeight = input_data.shape[0]
    imgWidth = input_data.shape[1]

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
    confidenceScores = [0] * numKeypoints
    for idx in range(len(keypointPositions)):
        position = keypointPositions[idx]
        positionY = keypointPositions[idx][0]
        positionX = keypointPositions[idx][1]
        # compute coordination and confidence value
        yCoords[idx] = int(positionY / (height - 1) *
                           imgHeight + offsets[0][positionY][positionX][idx])
        xCoords[idx] = int(positionX / (width - 1) * imgWidth +
                           offsets[0][positionY][positionX][idx + numKeypoints])
        confidenceScores[idx] = sigmoid(heatmaps[0][positionY][positionX][idx])

    result = {}
    for idx in range(len(keypointPositions)):
        result[PART_NAMES[idx]] = {
            'x': xCoords[idx],
            'y': yCoords[idx],
            'c': confidenceScores[idx]
        }

    print('Result:', json.dumps(result, indent=2))

    for key in result:
        point = result[key]['x'], result[key]['y']
        cv2.circle(input_data, point, 1, (0, 255, 0), 2)
    # render result
    for edge in CONNECTED_PART_NAMES:
        cv2.line(
            input_data,
            (result[edge[0]]['x'], result[edge[0]]['y']),
            (result[edge[1]]['x'], result[edge[1]]['y']),
            (0, 0, 255),
            2
        )

    cv2.imshow('frame', input_data)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
