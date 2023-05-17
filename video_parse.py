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
import proc_data_2 as proc2
from os.path import isfile

def main():

    overwrite = True
    play = False
    fbf = False         # only valid when play is True, only occurs on replays
    just_csv = False
    save_data = True
    db = False
    track_points = 5
    rsz = 0.85

    for q in range(26, 31):
        num = q
        target_data_path = 'opt_jump_' + str(num)# + '-1'
        target_data_path += '.npy'
        base_path = 'C:\\Users\\spenc\\Dropbox (MIT)\\2.671 Go Forth and Measure\\'
        proc2_data_path = 'C:\\Users\\spenc\\PycharmProjects\\2.671\\Proc2 Data Files\\'
        final_data_path = 'C:\\Users\\spenc\\PycharmProjects\\2.671\\Final Data Files\\'
        new_jump = 'new_jump\\mp4\\new_jump_' + str(num) + '.mp4'
        opt_jump = 'opt_jump\\mp4\\opt_jump_' + str(num) + '.mp4'
        final_jump = '\\Final\\mp4\\final_jump_' + str(num) + '.mp4'
        steven = 'Steven\\mp4\\steven' + str(num) + '.mp4'
        jackson = 'Jackson\\mp4\\jackson' + str(num) + '.mp4'

        chosen_path = base_path + final_jump
        data_path = final_data_path

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

            _Frame = ww.WindowWrapper('frame', targets=track_points, rsz_factor=rsz, fpath=chosen_path,
                 marker_buffer=0.03, hue_buffer=0.05, sat_buffer=0.15, val_buffer=0.15, visualize=True,
                 area_weight=0.5, color_weight=0, distance_weight=0.5, circularity_weight=0, filled_weight=0,
                 hyper=True, canny_thresh1=750, canny_thresh2=751, canny_apertureSize=5, canny_L2threshold=True,
                 error_threshold=0.5, debug=db, frame_by_frame=fbf)

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
            print(target_data_path + ' done!')
            key = ww.check()
            while key != ord('p') and key != ord('x'):
                key = ww.check()

            if key != ord('x'):
                _Frame.replay()

            _Frame.interpolate()
            data = _Frame.export_data()
            if save_data:
                np.save(data_path + target_data_path, data)
                df_data = proc2.process_struct(data)
                destination = data_path + target_data_path[:-4] + '.csv'
                df_data.to_csv(destination)
            else:
                print('--------- DATA SAVE SKIPPED ---------')
        else:
            print('NumPy file: ' + data_path + target_data_path + ' already exists!')
            data = np.load(data_path + target_data_path)
            destination = data_path + target_data_path[:-4] + '.csv'
            if not isfile(destination) and just_csv and save_data:
                df_data = proc2.process_struct(data)
                df_data.to_csv(destination)
            elif just_csv:
                print('You have chosen just_csv: True, but the file either already exists or you have chosen save_data: False.')
            else:
                print('CSV file: ' + data_path + target_data_path + ' already exists!')
            if play:
                _Frame = ww.WindowWrapper(fpath=chosen_path, rsz_factor=rsz, frame_by_frame=fbf)
                _Frame.set_data(data)
                _Frame.replay()
            '''
            data = np.load(data_path + target_data_path)
            interp_data = proc.angles_to_hor(data, [1])
            res = (interp_data[:, -1, 2:3] - interp_data[:, -1, 0:1]) % 360
            interp_data = np.hstack((interp_data, np.reshape(np.tile(res, (1, interp_data.shape[2])), (interp_data.shape[0], 1, interp_data.shape[2]))))
            if play:
                _Frame = ww.WindowWrapper(fpath=chosen_path, rsz_factor=rsz)
                _Frame.set_data(data)
                _Frame.replay()
            frames = np.reshape(np.arange(interp_data.shape[0]), (interp_data.shape[0], 1))
            export = np.hstack((frames, frames/240.0, interp_data[:, -1, 0:1]))
            export = pd.DataFrame(export, columns = ['Frame', 'Time (s)', 'Angle'])
            destination = data_path + target_data_path[:-4] + '.csv'
            if not isfile(destination):
                export.to_csv(destination)
            '''
if __name__ == '__main__':
    main()