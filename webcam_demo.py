import tensorflow as tf
tf = tf.compat.v1

import cv2
import time
import argparse
import csv
import posenet
import pickle 
from sklearn.metrics import pairwise_distances as distance

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=257)  # 1280
parser.add_argument('--cam_height', type=int, default=257)  # 720
parser.add_argument('--scale_factor', type=float, default=0.5)
parser.add_argument('--file', type=str, default=None,
                    help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

STAND_CRUNCH_LABELS = ["stand", "crunch"]
LANE_LABELS = ["left", "middle", "right"]
MIN_KEYPOINT_TO_PREDICT = 8

def extract_feature(keypoint_coords):
    features = distance(keypoint_coords[0:1], keypoint_coords[1:])[0]
    # normalize
    mx = max(features)
    mn = min(features)
    for index in range(len(features)):
        features[index] = (features[index] - mn) / (mx-mn)
    return features

def predict(model, keypoint_coords):
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
    
def main():
    with open("stand_crunch.model", "rb") as f:
        stand_crunch_model = pickle.load(f)

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        # if args.file is not None:
        #     cap = cv2.VideoCapture(args.file)
        #     flip = False
        # else:
        #     cap = cv2.VideoCapture(args.cam_id)
        flip = True

        cap = cv2.VideoCapture(0)
        # cap.set(3, args.cam_width)
        # cap.set(4, args.cam_height)
        scale_factor = 1/5
        start = time.time()
        frame_count = 0

        # header = ['action','frame', 'input_number', 'x_inputs', 'y_inputs']
        #
        # with open('dataset.csv','w') as w:
        #     do = csv.writer(w)
        #     do.writerow(header)

        # used to record the time when we processed last frame
        prev_frame_time = 0

        # used to record the time at which we processed current frame
        new_frame_time = 0

        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, flip=flip, scale_factor=scale_factor, output_stride=output_stride)

            # if not input_image:
            #     break
            # time when we finish processing for this frame
            new_frame_time = time.time()

            # Calculating the fps

            # fps will be number of frame processed in given time frame
            # since their will be most of time error of 0.001 second
            # we will be subtracting it to get more accurate result
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # converting the fps into integer
            fps = int(fps)

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
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            frame = []
            action_name = []
            position = []
            x_coordinates = []
            y_coordinates = []
            for i, j in enumerate(keypoint_coords[0]):
                action_name.append('walking')
                frame.append(frame_count)
                position.append(i + 1)
                x_coordinates.append(j[0])
                y_coordinates.append(j[1])
            
            # stand or crunch
            if len(keypoint_coords) > 0:
                posture_label = predict(stand_crunch_model, keypoint_coords[0])
                lane_label = get_lane(display_image, keypoint_coords[0])
                display_image = posenet.draw_string(display_image, posture_label + ", lane: " + lane_label)

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.15)

            cv2.imshow('posenet', overlay_image)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # print('Average FPS: ', frame_count / (time.time() - start))
            # print('Average FPS: ', fps)
            # rows = zip(action_name, frame, position,
            #            x_coordinates, y_coordinates)

            # with open('dataset.csv','a') as f:
            #     writer = csv.writer(f)
            #     #writer.writerow(header)
            #     for row in rows:
            #         writer.writerow(row)


if __name__ == "__main__":
    main()
