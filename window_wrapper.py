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

    def __init__(self, n, targets, rsz_factor=0.5, fpath='C:\\Users\\spenc\\Dropbox (MIT)\\2.671 Go Forth and Measure\\test.mp4',
                 marker_buffer=0.025, visualize=True):
        self.path = fpath
        self.name = n
        self.vis = visualize
        self.data_headers = ['Point Number', 'x', 'y', 'rx', 'ry', 'H', 'S', 'V', 'Error Code']
        self.pn = 0
        self.x = 1
        self.y = 2
        self.rx = 3
        self.ry = 4
        self.h = 5
        self.s = 6
        self.v = 7
        self.err = 8
        self.adv_struct = np.empty((1, len(self.data_headers)))
        self.trackers = targets
        self.frame = []
        self.retv = True

        self.contours = [[]] * self.trackers
        self.subframes = []
        self.corners = []
        self.tls = []
        self.centers = []
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
        if event == cv2.EVENT_LBUTTONDOWN:
            x_full = int((self.o_x/self.f_x)*x)
            y_full = int((self.o_y/self.f_y)*y)
            # noinspection PyTypeChecker
            temp_bgr = self.frame[y_full, x_full]
            temp_hsv = bgr2hsv(temp_bgr)
            np.append(self.adv_struct, (x, y, temp_hsv))
            np.append(self.adv_struct, (x_full, y_full, temp_hsv))
            cv2.circle(self.frame, (x_full, y_full), 2, (0, 0, 255), 2)
            cv2.imshow(self.name, cv2.resize(self.frame, (self.f_x, self.f_y)))

