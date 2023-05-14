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
from os.path import isfile

def main():

    num = 1
    target_data_path = 'new_jump_' + str(num)# + '-1'
    target_data_path += '.npy'
    base_path = 'C:\\Users\\spenc\\Dropbox (MIT)\\2.671 Go Forth and Measure\\'
    data_path = 'C:\\Users\\spenc\\PycharmProjects\\2.671\\New Data Files\\'
    new_jump = 'new_jump\\mp4\\new_jump_' + str(num) + '.mp4'
    opt_jump = 'opt_jump\\mp4\\opt_jump_' + str(num) + '.mp4'
    steven = 'Steven\\mp4\\steven' + str(num) + '.mp4'
    jackson = 'Jackson\\mp4\\jackson' + str(num) + '.mp4'

    overwrite = False
    play = True
    track_points = 4

    if not isfile(data_path + target_data_path) or overwrite:


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

        _Frame = ww.WindowWrapper('frame', targets=track_points, rsz_factor=0.5, fpath=base_path + opt_jump,
             marker_buffer=0.035, hue_buffer=0.075, sat_buffer=0.5, val_buffer=0.5, visualize=True,
             area_weight=0.75, color_weight=0, distance_weight=0.25, circularity_weight=0, filled_weight=0,
             hyper=True, canny_thresh1=750, canny_thresh2=751, canny_apertureSize=5, canny_L2threshold=True,
             error_threshold=0.5, debug=False)

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
        while key != ord('p') and key != ord('x'):
            key = ww.check()

        if key != ord('x'):
            _Frame.replay()

        data = _Frame.export_data()
        np.save(data_path + target_data_path, data)
    else:
        data_old = np.load(data_path + target_data_path)
        interp_data = proc.interpolate(data)
        interp_data = proc.angles_to_hor(data, [1])
        res = (interp_data[:, -1, 2:3] - interp_data[:, -1, 0:1]) % 360
        interp_data = np.hstack((interp_data, np.reshape(np.tile(res, (1, interp_data.shape[2])), (interp_data.shape[0], 1, interp_data.shape[2]))))
        if play:
            _Frame = ww.WindowWrapper(fpath=base_path+new_jump)
            _Frame.set_data(data_old)
            _Frame.replay()
        frames = np.reshape(np.arange(data.shape[0]), (data.shape[0], 1))
        export = np.hstack((frames, frames/240.0, data[:, -1, 0:1]))
        export = pd.DataFrame(export, columns = ['Frame', 'Time (s)', 'Angle'])
        destination = data_path + target_data_path[:-4] + '.csv'
        if not isfile(destination):
            export.to_csv(destination)
if __name__ == '__main__':
    main()