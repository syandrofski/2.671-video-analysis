import cv2
import copy
import numpy as np
import pandas as pd
import time
import math


def dist(x1, y1, x2, y2):
    #x1, y1 = c1
    #x2, y2 = c2
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


def hsv_distance(hsv1, hsv2):
    """Calculate the color space distance between two HSV points."""
    # Convert HSV values to 0-1 range
    hsv1_norm = hsv1 / np.array([180, 255, 255])
    hsv2_norm = hsv2 / np.array([180, 255, 255])

    # Calculate distance between two points
    distance = np.linalg.norm(hsv1_norm - hsv2_norm)

    return distance


class WindowWrapper:

    def __init__(self, n, targets=3, rsz_factor=0.5, fpath='C:\\Users\\spenc\\Dropbox (MIT)\\2.671 Go Forth and Measure\\test.mp4',
                 marker_buffer=0.025, hue_buffer=0.025, sat_buffer=0.7, val_buffer=0.7, visualize=True,
                 area_weight=0.334, color_weight=0.333, distance_weight=0.333):
        self.path = fpath
        self.name = n
        self.vis = visualize
        self.parsing = True

        self.data_headers = ['x', 'y', 'tlx', 'tly', 'rx', 'ry', 'px', 'py', 'H', 'S', 'V', 'Hull Area',
                             'Confidence', 'Buffer', 'Projection Type', 'Error Code']
        self.x = 0      # original frame x value
        self.y = 1      # original frame y value
        self.tlx = 2    # subframe top left x value
        self.tly = 3    # subframe top left y value
        self.rx = 4     # subframe relative x value
        self.ry = 5     # subframe relative y value
        self.px = 6     # projected absolute original frame x value for the next time step
        self.py = 7     # projected absolute original frame x value for the next time step
        self.h = 8     # hue value at current timestep
        self.s = 9     # saturation value at current timestep
        self.v = 10     # value value at current timestep
        self.hull = 11  # Hull area of the minimum enclosing hull of the best contour
        self.conf = 12  # confidence value describing the strength of the selected marker based on the previous frame
                        # conditions, specifically the linear projection and color value
        self.buf = 13   # subframe buffer size, only changes after obscurence
        self.pt = 14    # projection type, depending on point history (0-4)
        self.err = 15   # error code describing whether the confidence value was within an acceptable threshold, to
                        # predict obscurences

        self.trackers = targets
        self.adv_struct = np.zeros((1, len(self.data_headers), self.trackers))
        self.current = np.zeros((len(self.data_headers), self.trackers))
        self.last = np.zeros((len(self.data_headers), self.trackers))
        self.selections = 0

        self.oframe = []
        self.frame = []
        self.hsv = []
        self.canny = []
        self.retv = True
        self.replace = False
        self.i_tracker = -1

        self.contours = [[]] * self.trackers
        self.subframes = []
        self.sf_hsv = []
        self.sf_canny = []
        self.hyper = []
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

        self.h_buf = hue_buffer
        self.s_buf = sat_buffer
        self.v_buf = val_buffer

        self.area_wt = area_weight
        self.color_sim_wt = color_weight
        self.point_dist_wt = distance_weight

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

            self.adv_struct[0, self.x, self.selections] = x_full
            self.adv_struct[0, self.y, self.selections] = y_full
            self.adv_struct[0, self.rx, self.selections] = x
            self.adv_struct[0, self.ry, self.selections] = y
            self.adv_struct[0, self.h, self.selections] = temp_hsv[0]
            self.adv_struct[0, self.s, self.selections] = temp_hsv[1]
            self.adv_struct[0, self.v, self.selections] = temp_hsv[2]
            self.selections += 1

            cv2.circle(self.frame, (x_full, y_full), 2, (0, 0, 255), 2)
            cv2.imshow(self.name, cv2.resize(self.frame, (self.f_x, self.f_y)))
        elif event == cv2.EVENT_LBUTTONDOWN and self.replace is True:
            x_full, y_full = self.f2o((x, y))
            # noinspection PyTypeChecker
            temp_bgr = self.frame[y_full, x_full]
            temp_hsv = bgr2hsv(temp_bgr)

            self.current[self.x, self.i_tracker] = x_full
            self.current[self.y, self.i_tracker] = y_full
            self.current[self.px, self.i_tracker] = x_full
            self.current[self.py, self.i_tracker] = y_full
            self.current[self.rx, self.i_tracker] = x
            self.current[self.ry, self.i_tracker] = y
            self.current[self.h, self.i_tracker] = temp_hsv[0]
            self.current[self.s, self.i_tracker] = temp_hsv[1]
            self.current[self.v, self.i_tracker] = temp_hsv[2]

            self.i_tracker = -1
            self.replace = False

    def next_frame(self):
        self.retv, self.oframe = self.cap.read()
        if self.retv:
            self.f_num += 1

            self.last = np.copy(self.current)
            self.adv_struct = np.vstack((self.adv_struct, np.reshape(self.current, (1, self.current.shape[0], self.current.shape[1]))))
            self.current = np.zeros((len(self.data_headers), self.trackers))

        self.frame = copy.deepcopy(self.oframe)
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        self.canny = cv2.Canny(cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY), 750, 751, apertureSize=5, L2gradient=True)

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

    def subimage(self, thresholded=False):
        # Need to use oframes in the subimage method in order to avoid picking up the red marking dots in the subimages
        self.sf_num = 0

        self.subframes = []
        self.sf_hsv = []
        self.sf_canny = []
        self.hyper = []

        for i in range(self.trackers):
            self.current[self.pt, i] = self.assert_projection_type(i)

            self.project(i)

            if self.last[self.err, i] == 1:
                self.current[self.buf, i] = int(self.current[self.buf, i] * 1.1)
                if self.current[self.buf, i] >= self.o_y/2:
                    self.current[self.buf, i] = int(self.o_y/2.1)
                if self.current[self.buf, i] >= self.o_x/2:
                    self.current[self.buf, i] = int(self.o_x/2.1)

            self.current[self.tlx, i] = int(self.current[self.px, i] - self.current[self.buf, i])
            self.current[self.tly, i] = int(self.current[self.py, i] - self.current[self.buf, i])

            xmin = self.current[self.tlx, i]
            xmax = self.current[self.tlx, i] + int(2 * self.current[self.buf, i])
            ymin = self.current[self.tlx, i]
            ymax = self.current[self.tlx, i] + int(2 * self.current[self.buf, i])

            # For some reason for slicing, X and Y are switched and it's stupid
            self.subframes.append(self.oframe[ymin:ymax, xmin:xmax])
            temp_hsv = self.hsv[ymin:ymax, xmin:xmax]
            self.sf_hsv.append(temp_hsv)
            temp_canny = self.canny[ymin:ymax, xmin:xmax]
            self.sf_canny.append(temp_canny)

            temp_thresh = self.update_color_threshold(i)
            temp_range = cv2.inRange(temp_hsv, temp_thresh[0], temp_thresh[1])

            self.hyper.append(np.ceil((temp_range + temp_canny)/5))

    def update_color_threshold(self, i):
        temp_thresh = (self.last[self.h, i], self.last[self.s, i], self.last[self.v, i])
        h_noise = int(self.h_buf*180)
        s_noise = int(self.s_buf*256)
        v_noise = int(self.v_buf*256)
        bottom = np.array([temp_thresh[0]-h_noise, temp_thresh[1]-s_noise, temp_thresh[2]-v_noise])
        top = np.array([temp_thresh[0]+h_noise, temp_thresh[1]+s_noise, temp_thresh[2]+v_noise])
        if bottom[0] < 0:
            bottom[0] = 0
        if bottom[1] < 0:
            bottom[1] = 0
        if bottom[2] < 0:
            bottom[2] = 0
        if top[0] > 180:
            top[0] = 180
        if top[1] > 256:
            top[1] = 256
        if top[2] > 256:
            top[2] = 256
        return bottom, top

    def assert_projection_type(self, i):
        if self.f_num < 5:
            return 1
        if self.adv_struct[-1, self.pt, i] == 0 and self.adv_struct[-1, self.err, i] == 0:
            return 0
        err_codes = self.adv_struct[:, self.err, i]
        if np.sum(err_codes[-5:]) == 0:
            return 0
        if err_codes[-1] == 0 and err_codes[-2] == 0:
            return 1
        if np.sum(err_codes) == len(err_codes) - 1:
            return 4
        if np.sum(err_codes)/len(err_codes) < 0.5:
            return 2
        return 3

    def project(self, i):
        if self.adv_struct.shape[0] != 1:
            self.current[self.px, i] = self.last[self.x, i]
            self.current[self.py, i] = self.last[self.y, i]

        pt = self.current[self.pt, i]

        if pt == 0 or pt == 1:
            difference_x = np.diff(self.adv_struct[-2:, self.x, i].flatten())
            difference_y = np.diff(self.adv_struct[-2:, self.y, i].flatten())

            self.current[self.px, i] = self.last[self.x, i] + difference_x
            self.current[self.py, i] = self.last[self.y, i] + difference_y

            self.current[self.buf, i]  = self.m_buf
        elif pt == 2:
            err_codes = self.adv_struct[:, self.err, i]
            healthy = np.array(np.where(err_codes == 0))
            last = healthy[-1]
            stl = healthy[-2]
            span = abs(stl-last)

            difference_rate_x = (self.adv_struct[last, self.x, i] - self.adv_struct[stl, self.x, i])/span
            difference_rate_y = (self.adv_struct[last, self.y, i] - self.adv_struct[stl, self.y, i])/span

            proj_x = difference_rate_x * (self.f_num - 1 - last)
            proj_y = difference_rate_y * (self.f_num - 1 - last)

            self.current[self.px, i] = self.last[self.x, i] + proj_x
            self.current[self.py, i] = self.last[self.y, i] + proj_y

            self.current[self.buf, i] = self.m_buf * (1 + last/10.0)
        else: # pt == 3:
            self.i_tracker = i
            self.replace = True

            while self.replace:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    exit(1)
                elif key == ord('s'):
                    temp_pt = self.current[self.pt, i]
                    self.current = self.last
                    self.current[self.err, i] = 1
                    self.current[self.pt, i] = temp_pt
                    self.i_tracker = -1
                    self.replace = False

        self.repair_projection_bounds(i)

    def repair_projection_bounds(self, i):
        if self.current[self.px, i] <= 0:
            self.current[self.px, i] = 1
        if self.current[self.py, i] <= 0:
            self.current[self.py, i] = 1
        if self.current[self.px, i] >= self.o_x:
            self.current[self.px, i] = self.o_x - 1
        if self.current[self.py, i] >= self.o_y:
            self.current[self.py, i] = self.o_y - 1

    def repair_subframe_bounds(self, i):
        if self.current[self.tlx, i] <= 0:
            self.current[self.tlx, i] = 1
        if self.current[self.tly, i] <= 0:
            self.current[self.tly, i] = 1
        if self.current[self.tlx, i] >= self.o_x - self.current[self.buf, i]:
            self.current[self.tlx, i] = self.o_x - self.current[self.buf, i] - 1
        if self.current[self.tly, i] >= self.o_y - self.current[self.buf, i]:
            self.current[self.tly, i] = self.o_y - self.current[self.buf, i] - 1

    def contour_compare(self, contour):
        c, r = cv2. minEnclosingCircle(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        rc, (w, h), a = cv2.minAreaRect(contour)
        rect_area = w*h
        area_ratio = abs(math.log10(self.last[self.hull, self.i_tracker]/rect_area))
        # Again. reverse y and x to pull hsv from a numpy array
        hsv = self.sf_hsv[self.i_tracker][c[1], c[0]]
        last_hsv = np.array([self.last[self.h, self.i_tracker], self.last[self.s, self.i_tracker], self.last[self.s, self.i_tracker]])
        color_dist = 1 - hsv_distance(hsv, last_hsv)
        point_dist = dist(c[0], c[1], self.current[self.px, self.i_tracker], self.current[self.py, self.i_tracker])
        point_dist = point_dist / self.current[self.buf, self.i_tracker]
        if point_dist > 1:
            point_dist = 1.0
        point_dist = 1 - point_dist
        return (self.area_wt * area_ratio + self.color_sim_wt * color_dist + self.point_dist_wt * point_dist) / 3.0

    def analyze_subframe(self, i):
        self.contours[i], _ = cv2.findContours(self.hyper[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.i_tracker = i
        target_contour = max(self.contours[i], key=self.contour_compare)
        self.i_tracker = -1
        self.current[self.conf, i] = self.contour_compare(target_contour)
        (x, y), (w, h), a = cv2.minAreaRect(target_contour)
        self.current[self.hull, i] = w*h
        self.current[self.rx, i] = x
        self.current[self.ry, i] = y
        self.current[self.x, i] = x + self.current[self.tlx, i]
        self.current[self.y, i] = y + self.current[self.tly, i]
        if self.current[self.conf, i] < 0.5:
            self.current[self.err, i] = 1

    def draw(self):
        if self.vis:
            for i in range(self.trackers-1):
                cv2.line(self.frame, (self.current[self.x, i], self.current[self.y, i]),(self.current[self.x, i+1], self.current[self.y, i+1]), (0, 0, 255), 2)
            for j in range(self.trackers):
                if self.current[self.err, j] == 1:
                    marker_color = (0, 0, 255)
                    box_color = (0, 0, 255)
                else:
                    green = self.current[self.conf, j] * 255
                    red = 255 - green
                    marker_color = (0, green, red)
                    box_color = (150, 150, 150)
                cv2.rectangle(self.frame, (self.current[self.tlx, j], self.current[self.tly, j]),
                              (self.current[self.tlx, j] + 2*self.current[self.buf, j], self.current[self.tly, j] + 2*self.current[self.buf, j]),
                              box_color, 1)
                cv2.circle(self.frame, (self.current[self.x, j], self.current[self.y, j]), 2, marker_color, 2)
            cv2.imshow(self.name, cv2.resize(self.frame, (self.f_x, self.f_y)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                exit(0)

    def first_points(self):
        self.f_num = 1
        cv2.imshow(self.name, cv2.resize(self.frame, (self.f_x, self.f_y)))
        while self.selections < self.trackers:
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit(1)

        self.adv_struct = np.reshape(np.copy(self.current), (1, self.current.shape[0], self.current.shape[1]))

    def analyze_all_subframes(self):
        for i in range(self.trackers):
            self.analyze_subframes(i)