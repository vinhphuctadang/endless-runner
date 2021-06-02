import os
import cv2
import time
import posenet
import pandas as pd
import posenet.constants as consts
from tensorflow.compat.v1.keras.backend import get_session

SCORE_THRESHOLD = 0.15
TARGET_FILE = "stand.csv"
SOURCE_FILE = "/Users/dcongtinh/Workspace/endless-runner/pose/datasets/idle/Tinh_Idle_2.mov"
REWRITE = False
DRY_RUN = False # whether or not to write to file


def draw_skel_and_kp_showing_feature(display_image, keypoint_scores, keypoint_coords, min_part_score=0.15):
    origin = tuple(map(int, keypoint_coords[0]))
    for index in range(1, len(keypoint_scores)):
        if keypoint_scores[index] < min_part_score: # or keypoint_scores[indexes[1]] < min_part_score:
            if index == 0:
                break
            continue
        target = tuple(map(int, keypoint_coords[index]))
        display_image = cv2.line(display_image, (origin[1], origin[0]), (target[1], target[0]), (255, 0, 0), 1)

def draw_sparse_skel(display_image, keypoint_scores, keypoint_coords, min_part_score=0.1, point_per_edge = 5):
    # assert(type(point_per_edge) == "int" and point_per_edge > 0)
    # print(keypoint_scores)
    pairs = list(consts.CONNECTED_PART_INDICES)
    for x, y in pairs:
        if keypoint_scores[x] < min_part_score or keypoint_scores[y] < min_part_score:
            continue
        # generate points from coords[x] to coords[y]
        A, B = keypoint_coords[x], keypoint_coords[y]
        vector = ((B[0] - A[0])/point_per_edge, (B[1] - A[1])/point_per_edge)
        for i in range(point_per_edge + 1):
            C = (int(A[0] + vector[0] * i), int(A[1] + vector[1] * i))
            display_image = cv2.circle(display_image, (C[1], C[0]), 3, color=(0, 255, 0), thickness=3)

    for partId in range(5): # 5 first parts containng nose, left, right eyes, ears
        # draw nose as well
        if keypoint_scores[partId] < min_part_score:
            continue
        C = keypoint_coords[partId]
        display_image = cv2.circle(display_image, (int(C[1]), int(C[0])), 3, color=(0, 255, 0), thickness=3)

    return display_image

def main():
    model = 101
    sess = get_session()
    model_cfg, model_outputs = posenet.load_model(model, sess)
    output_stride = model_cfg['output_stride']

    # flip the video
    flip = True

    # re-scale for faster detection
    scale_factor = 0.4

    source_name = SOURCE_FILE
    # set capture source
    cap = cv2.VideoCapture(source_name)

    # split file name
    target_file = TARGET_FILE

    # start = time.time()
    frame_count = 0

    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

    # df
    if not REWRITE and os.path.isfile(target_file):
        df = pd.read_csv(target_file, index_col=0)
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
            min_pose_score=SCORE_THRESHOLD, min_part_score=SCORE_THRESHOLD)
        # draw_sparse_skel(display_image, keypoint_scores[0], keypoint_coords[0], min_part_score=SCORE_THRESHOLD)

        # save key points to data frame:
        # flatten
        keypoint_coords_list = []
        for coord in keypoint_coords[0]: keypoint_coords_list.append(list(coord))
        keypoint_scores_list = list(keypoint_scores[0])

        # print("keypoints:", keypoint_coords_flattened)
        # append to dataframe (ignoring index)

        if not DRY_RUN:
            df = df.append(dict(zip(df.columns, [keypoint_coords_list, keypoint_scores_list])), ignore_index=True)

        cv2.imshow('posenet', display_image)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if not DRY_RUN:
        df.to_csv(target_file)

if __name__ == "__main__":
    main()
