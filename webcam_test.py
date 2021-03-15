#
# Vinh Phuc Ta Dang ft Dao Cong Tinh
#
import os
import cv2
import posenet
import numpy as np
from constants import *
from scipy.spatial.distance import euclidean
from tensorflow.keras.models import load_model
from tensorflow.compat.v1.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

VID_EXT_VALIDS = ['.mp4', '.mov']
scale_factor = 0.5
GUI_Enable = True
VIDEO_URI = '/Users/dcongtinh/Workspace/endless-runner/pose/datasets/running/Tinh_Running_2.mov'
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale, thickness = 0.75, 2

actions = np.load('labels.npy')
actions = np.unique(actions)
model_loaded = load_model(
    '/Users/dcongtinh/Workspace/endless-runner/results/20210314_160244/20210314_160244model.h5')


def main():
    sess = K.get_session()
    model_cfg, model_outputs = posenet.load_model(101, sess)
    output_stride = model_cfg['output_stride']

    cap = cv2.VideoCapture(VIDEO_URI)
    if type(VIDEO_URI) == type(''):
        flip = False
    else:
        flip = True
    cap.set(3, 257)
    cap.set(4, 257)
    frame_count = 0
    frame_series, frame_seq = [], []
    label = 'idle'

    print('Extracting video ' + str(VIDEO_URI) + '...')
    while True:
        try:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, flip=flip, scale_factor=scale_factor, output_stride=output_stride)
        except:
            print("End video %s.%d\n" % (VIDEO_URI, len(frame_series)))
            break

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
            min_pose_score=0.15)

        keypoint_coords *= output_scale
        first_keypoint = keypoint_coords[0]
        nose = first_keypoint[0]
        feature = []
        for i in range(1, len(first_keypoint)):
            feature.append(euclidean(nose, first_keypoint[i]))

        frame_count += 1
        frame_seq.append(feature)
        if frame_count % window_size == 0:
            frame_seq = np.array(frame_seq)
            frame_seq = np.expand_dims(frame_seq, axis=0)
            pred = model_loaded.predict(frame_seq)[0]
            pred = np.argmax(pred)
            label = actions[pred]
            frame_seq = []

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        if GUI_Enable:
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            cv2.putText(overlay_image, label, (20, 20),
                        fontFace, fontScale=fontScale, color=(0, 255, 0), thickness=thickness)
            cv2.imshow('posenet', overlay_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
