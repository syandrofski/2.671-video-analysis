import sys
import cv2
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib as plt
from math import sin, cos, tan, atan, sqrt
import time
import copy
import window_wrapper as ww
import proc_data as proc


def main():

    base_path = 'C:\\Users\\spenc\\Dropbox (MIT)\\2.671 Go Forth and Measure\\'
    num = 7
    jump = 'jump\\mp4\\jump' + str(num) + '.mp4'
    steven = 'Steven\\mp4\\steven' + str(num) + '.mp4'
    jackson = 'Jackson\\mp4\\jackson' + str(num) + '.mp4'

    track_points = 3

    # Add in a slider, effectively, dividing saturation and value
    # Smart dynamic thresholding?
    # Detect all points first, then sort by comparing to previous points
    # Keeps last confidence score for point
    # Replays from pandas array rather than recalculating every time

    # Size similarities to previous contour

    # Add linear interpolation for unknown regions? in post?

    ''' Frame tester
    cv2.imshow('test_frame', self.subframes[i])
    key = cv2.waitKey(1) & 0xFF
    while key != ord('q'):
        key = cv2.waitKey(1) & 0xFF
    exit(99)
    '''

    _Frame = ww.WindowWrapper('frame', targets=track_points, rsz_factor=0.6, fpath=base_path + jump,
             marker_buffer=0.025, hue_buffer=0.025, sat_buffer=0.5, val_buffer=0.5, visualize=True,
             area_weight=0.3, color_weight=0.2, distance_weight=0.2, circularity_weight=0.3, filled_weight=0,
             hyper=False, canny_thresh1=750, canny_thresh2=751, canny_apertureSize=5, canny_L2threshold=True, debug=False)

    first_frame = True
    retv = _Frame.retv

    while retv:
        # Capture a frame from the webcam
        _Frame.next_frame()
        retv = _Frame.retv

        if first_frame:
            _Frame.get_first_points()
            first_frame = False

        elif retv:
            _Frame.pre_subimage_project()
            _Frame.subimage()

            _Frame.analyze_all_subframes()

            _Frame.draw()

            # Quit if user presses q
            key = ww.check()

            '''
            # Quit if user presses q, next frame if user presses n
            while True:
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    exit(0)
                elif key & 0xFF == ord('n'):
                    break
            '''

        else:
            # Quit if user presses q
            key = ww.check()
    print('done!')
    key = ww.check()
    while key != ord('p'):
        key = ww.check()

    _Frame.replay()

    data = _Frame.export_data()
    proc.angles_to_hor(data)
if __name__ == '__main__':
    main()