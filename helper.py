import cv2
import numpy as np
import mediapipe as mp

from mediapipe.framework.formats import landmark_pb2

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_RGB_CHANNELS = 3


def hands_in_box_area_func(landmark_list: landmark_pb2.NormalizedLandmarkList, width:int, height:int, top_left, bot_right):
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
            landmark.visibility < _VISIBILITY_THRESHOLD) or
            (landmark.HasField('presence') and
            landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                    width, height)
        if top_left[0] < landmark_px[0]  < bot_right[0] and top_left[1] < landmark_px[1]  < bot_right[1]:
            return True
    return False


def item_in_box_wrapper(hsv_seg_func, mask,  threshold=0.5):
    def item_in_box_area(new_im):
        # if empty_area_overlap_ratio less than threshold value, then 
        new_mask, _ = hsv_seg_func(new_im)
        cv2.imshow('Two masks', np.concatenate([mask, new_mask], axis=1))
        empty_area_overlap_ratio = np.sum(cv2.bitwise_and(mask, new_mask))/np.sum(mask)
        print(f'Overlapping ratio: {empty_area_overlap_ratio}', end='\r')
        return empty_area_overlap_ratio # < threshold
    return item_in_box_area