import os
import sys
import math
import numpy as np
import cv2

# --------------------config-------------------- #

# Models
#DET_MODEL = 'rtmdet-l' # Detection model
#POSE_MODEL = 'rtmpose-l_8xb512-700e_body8-halpe26-384x288' # Pose model
POSE_CONFIG = './configs/rtmpose_l_26.py'
POSE_CHECKPOINT = './weights/rtmpose_l_26.pth'
TRACK_CONFIG = './configs/ocsort.py'
DET_CHECKPOINT = './weights/rtmdet_l.pth'

TRACK_THRESH = 0.5

# --------------------end config-------------------- #


def pose_generation(filepath_in: str):

    # bounding boxes

    # tracking oc-sort

    # pose estimation

    # smoothing


    results_2d, results_3d = ..., ...  # Placeholder for actual pose generation logic
    return results_2d, results_3d    