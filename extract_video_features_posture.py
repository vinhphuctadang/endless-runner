import os
import cv2
import time
import posenet
import pandas as pd
from tensorflow.compat.v1.keras import backend as K

SCORE_THRESHOLD = 0.15

# PATH to CSV file which will contain extracted posture
TARGET_FILE = "run.csv"

# PATH TO SOURCE file
SOURCE_FILE = "/Users/dcongtinh/Workspace/endless-runner/pose/datasets/running/Tinh_Running_v2.2.mov"

# Rewrite or Append to csv file
REWRITE = False

def draw_skel_and_kp_showing_feature(display_image, keypoint_scores, keypoint_coords, min_part_score=0.1):
    origin = tuple(map(int, keypoint_coords[0]))
    for index in range(1, len(keypoint_scores)):
        if keypoint_scores[index] < min_part_score: # or keypoint_scores[indexes[1]] < min_part_score:
            if index == 0:
                break
            continue
        target = tuple(map(int, keypoint_coords[index]))
        display_image = cv2.line(display_image, (origin[1], origin[0]), (target[1], target[0]), (255, 0, 0), 1)

def main():
    # model number
    model = 101
    
    # init session
    sess = K.get_session()

    model_cfg, model_outputs = posenet.load_model(model, sess)
    output_stride = model_cfg['output_stride']
    # flip the video
    flip = False
    # re-scale for faster detection
    scale_factor = 1/6

    # set capture source
    cap = cv2.VideoCapture(SOURCE_FILE)
    frame_count = 0

    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

    # df
    if not REWRITE and os.path.isfile(TARGET_FILE):
        df = pd.read_csv(TARGET_FILE, index_col=0)
    else:
        df = pd.DataFrame({"keypoint_coords": [], "keypoint_scores": []})

    while True:
        try:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, flip=flip, scale_factor=scale_factor, output_stride=output_stride)
        except Exception as err:
            print("Error: Read cap. Details:", err)
            break

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
            output_stride       = output_stride,
            max_pose_detections = 1,
            min_pose_score      = SCORE_THRESHOLD)

        # original_coords = keypoint_coords
        keypoint_coords *= output_scale

        # print("keypoints:", keypoint_coords)

        display_image = posenet.draw_fps(display_image, fps)
        display_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=SCORE_THRESHOLD)
        draw_skel_and_kp_showing_feature(display_image, keypoint_scores[0], keypoint_coords[0], min_part_score=SCORE_THRESHOLD)

        # save key points to data frame:
        # flatten
        keypoint_coords_list = []
        for coord in keypoint_coords[0]: keypoint_coords_list.append(list(coord))
        keypoint_scores_list = list(keypoint_scores[0])

        # print("keypoints:", keypoint_coords_flattened)
        # append to dataframe (ignoring index)
        df = df.append(dict(zip(df.columns, [keypoint_coords_list, keypoint_scores_list])), ignore_index=True)

        cv2.imshow('posenet', display_image)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    df.to_csv(TARGET_FILE)

if __name__ == "__main__":
    main()
