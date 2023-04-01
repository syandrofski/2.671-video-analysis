import cv2
import copy
import numpy as np
import pandas as pd
import time
import math


def dist(c1, c2):
    x1, y1 = c1
    x2, y2 = c2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def regularity(contour, rect=False):
    if rect:
        hull = cv2.contourArea(cv2.convexHull(contour))
        _, (w, h), _ = cv2.minAreaRect(contour)
        if w == 0 or h == 0:
            res = -1
        else:
            res = hull / (w * h)
        return res
    hull = cv2.contourArea(cv2.convexHull(contour))
    _, r = cv2.minEnclosingCircle(contour)
    circle = 3.1415 * r ** 2
    res = -abs(hull / circle - 1)
    if r < 3:
        res = -10
    return res


def marker_sort(mkr):
    return mkr[1]


def bgr2hsv(in_px):
    in_px = in_px/255.0
    b = in_px[0]
    g = in_px[1]
    r = in_px[2]
    cmax = max(b, g, r)
    cmin = min(b, g, r)
    delta = cmax - cmin

    # Hue calculation
    if delta == 0:
        hue = 0
    elif cmax == r:
        hue = ((g - b) / delta) % 6
    elif cmax == g:
        hue = (b - r) / delta + 2
    else:
        hue = (r - g) / delta + 4
    hue = round(hue * 30)

    # Saturation calculation
    if cmax == 0:
        saturation = 0
    else:
        saturation = delta / cmax
    saturation = round(saturation * 255)

    # Value calculation
    value = round(cmax * 255)

    return np.array([hue, saturation, value])


class WindowWrapper:

    def __init__(self, n, targets=3, rsz_factor=0.5, fpath='C:\\Users\\spenc\\Dropbox (MIT)\\2.671 Go Forth and Measure\\test.mp4',
                 marker_buffer=0.025, visualize=True):
        self.path = fpath
        self.name = n
        self.vis = visualize

        self.data_headers = ['x', 'y', 'rx', 'ry', 'px', 'py', 'prx', 'pry', 'H', 'S', 'V', 'Confidence', 'Buffer', 'Error Code']
        self.x = 0      # original frame x value
        self.y = 1      # original frame y value
        self.tlx = 2    # subframe top left x value
        self.tly = 3    # subframe top left y value
        self.rx = 4     # subframe relative x value
        self.ry = 5     # subframe relative y value
        self.px = 6     # projected absolute original frame x value for the next time step
        self.py = 7     # projected absolute original frame x value for the next time step
        self.prx = 8    # projected subframe x value, relative to the current timestep tlx
        self.pry = 9    # projected subframe y value, relative to the current timestep tly
        self.h = 10     # hue value at current timestep
        self.s = 11     # saturation value at current timestep
        self.v = 12     # value value at current timestep
        self.conf = 13  # confidence value describing the strength of the selected marker based on the previous frame
                        # conditions, specifically the linear projection and color value
        self.buf = 14   # subframe buffer size, only changes after obscurence
        self.err = 15   # error code describing whether the confidence value was within an acceptable threshold, to
                        # predict obscurences

        self.trackers = targets
        self.adv_struct = np.zeros((1, len(self.data_headers), self.trackers))
        self.current = np.zeros((1, len(self.data_headers), self.trackers))
        self.selections = 0

        self.oframe = []
        self.frame = []
        self.retv = True

        self.contours = [[]] * self.trackers
        self.subframes = []
        self.rsz = rsz_factor

        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self.on_click)

        # cap is a VideoCapture object we can use to get frames from video files
        self.cap = cv2.VideoCapture(self.path)

        # Gets an initial frame
        self.next_frame()

        # Resets capture file
        self.cap = cv2.VideoCapture(self.path)
        self.f_num = 0
        self.sf_num = 0

        self.o_x = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.o_y = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.f_x = int(self.rsz * self.o_x)
        self.f_y = int(self.rsz * self.o_y)

        if marker_buffer < 1:
            if self.o_x < self.o_y:
                self.m_buf = int(self.o_x * marker_buffer)
            else:
                self.m_buf = int(self.o_y * marker_buffer)
        else:
            self.m_buf = int(marker_buffer)

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.selections < self.trackers:
            x_full, y_full = self.f2o((x, y))
            # noinspection PyTypeChecker
            temp_bgr = self.frame[y_full, x_full]
            temp_hsv = bgr2hsv(temp_bgr)

            self.current[0, self.x, self.selections] = x_full
            self.current[0, self.y, self.selections] = y_full
            self.current[0, self.rx, self.selections] = x
            self.current[0, self.ry, self.selections] = y
            self.current[0, self.h, self.selections] = temp_hsv[0]
            self.current[0, self.s, self.selections] = temp_hsv[1]
            self.current[0, self.v, self.selections] = temp_hsv[2]
            self.selections += 1

            cv2.circle(self.frame, (x_full, y_full), 2, (0, 0, 255), 2)
            cv2.imshow(self.name, cv2.resize(self.frame, (self.f_x, self.f_y)))

    def next_frame(self):
        self.retv, self.oframe = self.cap.read()
        if self.retv:
            self.f_num += 1
        self.frame = copy.deepcopy(self.oframe)

    def o2f(self, xy):
        x, y = xy
        return int(x*self.rsz), int(y*self.rsz)

    def f2o(self, xy):
        x, y = xy
        return int(x/self.rsz), int(y/self.rsz)

    def sf2f(self, tl, rxy):
        tlx, tly = tl
        rx, ry = rxy
        return int(tlx+rx), int(tly+ry)

    def sf2o(self, tl, rxy):
        return self.f2o(self.sf2f(tl, rxy))

    def subimage(self, thresholded = False):
        # Need to use oframes in the subimage method in order to avoid picking up the red marking dots in the subimages
        self.sf_num = 0

        self.corners = []
        self.subframes = []
        self.tls = []
        h = self.o_x
        w = self.o_y

        for entry in self.current:

            if self.current # if the current point is an error point, up the buffer by 10% and resave. That way, the buffer looks forward one frame.
            buffer = self.m_buf
            try:
                tl_x = center[0] - self.m_buf
            except:
                tl_x = 0

            try:
                tl_y = center[1] - self.m_buf
            except:
                tl_y = 0

            try:
                br_x = center[0] + self.m_buf
            except:
                br_x = h - 1

            try:
                br_y = center[1] + self.m_buf
            except:
                br_y = w - 1

            # X positive moving right, Y positive moving down, (0, 0) in top left
            self.corners.append([(tl_x, tl_y), (br_x, br_y)])

            self.tls.append([tl_x, tl_y])

        for pair in self.corners:
            # For some reason for slicing, X and Y are switched and it's stupid
            self.subframes.append(self.oframe[pair[0][1]:pair[1][1], pair[0][0]:pair[1][0]])
        return self.corners, self.subframes, self.tls