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


def main():

    base_path = 'C:\\Users\\spenc\\Dropbox (MIT)\\2.671 Go Forth and Measure\\'
    num = 1
    t1 = 'jump1_AdobeExpress.mp4'
    steven = 'Steven\\mp4\\steven' + str(num) + '.mp4'
    jackson = 'Jackson\\mp4\\jackson' + str(num) + '.mp4'

    first_view = True
    i_markers = []

    while True:

        track_points = 3

        # Add in a slider, effectively, dividing saturation and value
        # Smart dynamic thresholding?
        #Detect all points first, then sort by comparing to previous points
        # Keeps last confidence score for point
        # Replays from pandas array rather than recalculating every time

        _Frame = ww.WindowWrapper('frame', track_points,
                                  fpath=base_path + jackson,
                                  marker_buffer=0.02, rsz_factor=0.5, init_frms=i_markers, hue_buffer=0.02, sat_val_buffer=0.35,
                                  proximity_weight=0, testing=False, auto_color=True, data_output=True)

        first_frame = True
        retv = _Frame.get_retrieval_state()

        while retv:
            # Capture a frame from the webcam
            retv, oframe, frame = _Frame.next_frame()

            if retv:
                _Frame.show()
                time.sleep(5)
                exit(4)

            if retv:

                corners, frames, tls = _Frame.subimage(first_view, first_frame)

                for i, frame in enumerate(frames):
                    if not first_frame:
                        _Frame.update_contours()
                    _Frame.analyze_contours(first_view, first_frame)

                # Quit if user presses q
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    exit(0)

                '''
                # Quit if user presses q, next frame if user presses n
                while True:
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        exit(0)
                    elif key & 0xFF == ord('n'):
                        break
                '''
                _Frame.show()

            else:
                # Quit if u#ser presses q
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    exit(0)

            first_frame = False

        i_markers = _Frame.get_initial_markers()

        if first_view:
            print(_Frame.get_data())

        first_view = False

        key = cv2.waitKey(1) & 0xFF
        while key != ord('q') and key != ord('p'):
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                exit(0)


if __name__ == '__main__':
    main()

'''
def track(fpath, color, points, reg='circ'):
    if reg == 'circ':
        reg_fn = regularity_c
    else:
        reg_fn = regularity

    track_points = points
    all_points = []
    for i in range(points):
        all_points.append([])

    if 'C:' not in fpath:
        fpath = 'C:\\Users\\spenc\\Dropbox (MIT)\\2.671 Go Forth and Measure\\' + fpath

    markers = [(0, 0)] * track_points
    r_markers = [(0, 0)] * track_points

    # Min and max HSV thresholds for target color
    if color == 'w':
        COLOR_THRESHOLD = ([0, 0, 180], [60, 60, 255])  # White
    else:
        COLOR_THRESHOLD = ([20, 100, 100], [35, 255, 255])  # Yellow

    # cap is a VideoCapture object we can use to get frames from webcam
    cap = cv2.VideoCapture(fpath)

    first = True
    retv = True

    #fnum = 0

    while retv:
        # Capture a frame from the webcam
        retv, frame = cap.read()
        #print('frame', fnum)
        #fnum += 1

        if retv:
            f_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            f_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            if not first:
                corners, frames, tls = subimage(frame, markers)
            else:
                corners, frames, tls = subimage_gen_empty(frame, markers)

            for i, frame in enumerate(frames):
                # Convert BGR to HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # OpenCV needs bounds as numpy arrays
                lower_color_bound = np.array(COLOR_THRESHOLD[0])
                upper_color_bound = np.array(COLOR_THRESHOLD[1])

                mask_color = cv2.inRange(hsv, lower_color_bound, upper_color_bound)
                contours_color, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # If we have contours...
                if len(contours_color) != 0:

                    # Find the biggest contour by area
                    #g = max(contours_color, key=cv2.contourArea)

                    if first:
                        # Find the most circular contour by hull area
                        cc_list = list(contours_color)
                        cc_list.sort(reverse=True, key=reg_fn)
                        contours_color = cc_list
                        #contours_color = [max(contours_color, key=regularity)]
                        max_significant = 3
                        for j in range(max_significant):
                            circle = contours_color[j]

                            # Get a bounding rectangle around that contour
                            #x, y, w, h = cv2.boundingRect(circle)
                            c, r = cv2.minEnclosingCircle(circle)

                            markers[j] = (int(c[0]), int(c[1]))
                            r_markers[j] = (int(c[0]), int(c[1]))

                            all_points[j].append(markers[j])
                    else:
                        contours_color = max(contours_color, key=reg_fn)
                        c, r = cv2.minEnclosingCircle(contours_color)

                        r_markers[i] = (int(c[0])+1, int(c[1])+1)
                        markers[i] = (tls[i][0]+r_markers[i][0], tls[i][1]+r_markers[i][1])

                        all_points[i].append(markers[i])

        first = False
    df_x = []
    df_y = []
    for single_point in all_points:
        sparr = np.array(single_point)
        df_x.append(sparr[:, 0:1])
        df_y.append(sparr[:, 1:2])
    df_x = tuple(df_x)
    df_y = tuple(df_y)
    df_x = pd.DataFrame(np.hstack(df_x))
    df_y = pd.DataFrame(np.hstack(df_y))
    return df_x, df_y
'''

'''
                    # If we have contours...
                    print('frame')
                    if len(contours) != 0:
                        print('contours!')

                        # Find the biggest contour by area
                        #g = max(contours_color, key=cv2.contourArea)

                        if redo and first:
                            markers = _Frame.set_initial_points()
                        elif first:
                            if auto_select:
                                # Find the most circular contour by hull area
                                cc_list = list(contours_color)
                                cc_list.sort(reverse=True, key=regularity_c)
                                contours_color = cc_list
                                #contours_color = [max(contours_color, key=regularity)]
                                max_significant = 3
                                for j in range(max_significant):
                                    circle = contours_color[j]

                                    # Get a bounding rectangle around that contour
                                    #x, y, w, h = cv2.boundingRect(circle)
                                    c, r = cv2.minEnclosingCircle(circle)

                                    markers[j] = (int(c[0]), int(c[1]))
                                    r_markers[j] = (int(c[0]), int(c[1]))

                                    # Draw the rectangle on our frame
                                    #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 208, 255), 2)

                                    #Draw the circle on the frame
                                    cv2.circle(oframe, (markers[j][0], markers[j][1]), int(r), (0, 255, 0), 2)

                                    #multiline used to start here Draws lines in the first frame, sometimes causes big skips
                                    for k, pt in enumerate(markers):
                                        if k < len(markers)-1:
                                            cv2.line(oframe, (markers[k][0], markers[k][1]), (markers[k+1][0], markers[k+1][1]), (0, 0, 255), 3)
                                    #and end here
                            else:
                                cv2.imshow('frame', cv2.resize(oframe, (adj_x, adj_y)))
                                while len(wrapped_window.get_markers()) < track_points:
                                    key = cv2.waitKey(1) & 0xFF
                                    if key == ord('q'):
                                        exit(1)
                                markers = wrapped_window.get_markers()
                                first_markers = wrapped_window.get_markers()

                        else:
                            contours_color = max(contours_color, key=regularity_c)
                            c, r = cv2.minEnclosingCircle(contours_color)
                            print(c, r)

                            r_markers[i] = (int(c[0])+1, int(c[1])+1)
                            markers[i] = (tls[i][0]+r_markers[i][0], tls[i][1]+r_markers[i][1])

                            #Draw the circle on the frame
                            cv2.circle(oframe, (markers[i][0], markers[i][1]), int(r), (0, 255, 0), 2)

                            #Draw framing rectangles
                            cv2.rectangle(oframe, (corners[i][0][0], corners[i][0][1]),
                                          (corners[i][1][0], corners[i][1][1]), (255, 0, 0), 2)
                            cv2.circle(oframe, (corners[i][0][0], corners[i][0][1]), 10, (255, 0, 0), 2)

                            for k, pt in enumerate(markers):
                                if k < len(markers)-1:
                                    cv2.line(oframe, (markers[k][0], markers[k][1]), (markers[k+1][0], markers[k+1][1]), (0, 0, 255), 5)

                    # Display that frame (resized to be smaller for convenience)
                    #cv2.imshow('color mask', cv2.resize(mask_color, (adj_x, adj_y)))

                    #if not first:
                    #    cv2.circle(frame, (r_markers[i][0], r_markers[i][1]), 1, (0, 0, 255), 3)
                    #    cv2.imshow('subframe', frame)

                    #multiline used to start here
                    lines = []
                    for i, center in enumerate(markers):
                        if not i == len(markers)-1:
                            lines.append(((markers[i][0], markers[i][1]), (markers[i+1][0], markers[i+1][1])))
                    cv2.drawContours(oframe, lines, (0, 0, 255), 5)
                    # and end here

                    if not (auto_select and first):
                        cv2.imshow('frame', cv2.resize(oframe, (adj_x, adj_y)))
                    redo = True
                    #print(markers)
'''

'''


def regularity(contour, tp='rect'):
    if tp == 'rect':
        hull = cv2.contourArea(cv2.convexHull(contour))
        (x, y), (w, h), _ = cv2.minAreaRect(contour)
        if w == 0 or h == 0:
            res = -1
        else:
            res = hull/(w*h)
    else:
        hull = cv2.contourArea(cv2.convexHull(contour))
        _, r = cv2.minEnclosingCircle(contour)
        circ = 3.1415*r**2
        res = -abs(hull/circ - 1)
        if r < 3:
            res = -10
    return res


def regularity_c(contour):
    return regularity(contour, tp='circ')


def subimage_gen_empty(frame, centers):
    corners = []
    tls = []
    for center in centers:
        tl_x = 0
        tl_y = 0
        br_x = 100
        br_y = 100
        # X positive moving right, Y positive moving down, (0, 0) in top left
        corners.append([(tl_x, tl_y), (br_x, br_y)])
        tls.append([tl_x, tl_y])
    subframes = [frame]
    return corners, subframes, tls


def auto_select(frame, contours, markers, rel_markers):
    if auto_select:
        # Find the most circular contour by hull area
        cc_list = list(contours)
        cc_list.sort(reverse=True, key=regularity_c)
        contours = cc_list
        # contours_color = [max(contours_color, key=regularity)]
        max_significant = len(markers)
        for j in range(max_significant):
            circle = contours[j]

            # Get a bounding rectangle around that contour
            # x, y, w, h = cv2.boundingRect(circle)
            c, r = cv2.minEnclosingCircle(circle)

            markers[j] = (int(c[0]), int(c[1]))
            rel_markers[j] = (int(c[0]), int(c[1]))

            # Draw the rectangle on our frame
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 208, 255), 2)

            # Draw the circle on the frame
            cv2.circle(frame, (markers[j][0], markers[j][1]), int(r), (0, 255, 0), 2)
'''