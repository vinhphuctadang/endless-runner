#
# Vinh Phuc Ta Dang 
#
import tensorflow as tf
import cv2 
import numpy as np
import json

MODEL_PATH = 'model/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
DST_SIZE = (257, 257)

# # parts of pose
# # below constants referred from https://github.com/rwightman/posenet-python
# # special thanks for the repo authors
# PART_NAMES = [
#     "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
#     "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
#     "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
# ]

# CONNECTED_PART_NAMES = [
#     ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
#     ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
#     ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
#     ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
#     ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
#     ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
# ]

# Doing the legend steps:
# tf.lite.Interpreter() -> transform data -> run inferences -> Interprete output

def main():
    
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    print('Loading image ...')
    input_data = cv2.imread('./pose_scaled.png')
    # input_data = cv2.resize(input_data, DST_SIZE) # scale image

    input_data = input_data.astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], [input_data])
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Output data is now a heatmap
    print('Output:', output_data)
    
if __name__ == '__main__':
    main()