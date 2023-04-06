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

    def __init__(self, n, targets, init_frms=(), rsz_factor=0.5, fpath='C:\\Users\\spenc\\Dropbox (MIT)\\2.671 Go Forth and Measure\\test.mp4',
                 marker_buffer=0.025, hue_buffer=0.05, sat_val_buffer=0.5, testing=False, auto_select=False, auto_color=False, visualize=True,
                 data_output=False, proximity_weight=0.5):
        self.path = fpath
        self.name = n
        self.vis = visualize
        self.d_out = data_output
        self.data_headers = ['Point Number', 'x', 'y', 'H', 'S', 'V', 'Error Code']
        self.data = np.empty((1, len(self.data_headers)))
        self.f_num = 0
        self.trackers = targets

        # Formally initialized later in constructor
        self.oframe = []
        self.frame = []
        self.retv = True
        self.hsv = []
        self.full_hsv = []
        #self.color_mask = []
        self.color_masks = [[]] * self.trackers
        self.i = 0

        self.contours = [[]] * self.trackers
        self.subframes = []
        self.corners = []
        self.tls = []
        self.rsz = rsz_factor

        self.current_relative_marker = []
        self.markers = []
        if len(init_frms) == 0:
            self.initial_markers = []
        else:
            self.initial_markers = init_frms
        self.relative_markers = copy.deepcopy(self.initial_markers)

        self._test = testing
        if auto_select and auto_color:
            print('Auto-selection and auto-color detection are incompatible.\nAuto-color detection has been disabled to '
                  'preserve expected automatic behavior.')
            self._as = True
            self._ac = False
        else:
            self._as = auto_select
            self._ac = auto_color

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

        self.h_buf = hue_buffer
        self.sv_buf = sat_val_buffer
        self.prox_wt = proximity_weight
        if marker_buffer < 1:
            if self.o_x < self.o_y:
                self.m_buf = int(self.o_x * marker_buffer)
            else:
                self.m_buf = int(self.o_y * marker_buffer)
        else:
            self.m_buf = int(marker_buffer)

        # Min and max HSV thresholds for target color
        self.thresh = (np.array([0, 0, 0]), np.array([179, 255, 255]))

        if not self._ac:
            if self._test:
                self.thresh = (np.array([20, 100, 100]), np.array([35, 255, 255]))  # Yellow
            else:
                self.thresh = (np.array([0, 0, 180]), np.array([60, 60, 255]))  # White
            self.lcb = self.thresh[0]
            self.ucb = self.thresh[1]
            self.thresh = [self.thresh]
        else:
            self.thresh = [self.thresh] * self.trackers

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x_full = int((self.o_x/self.f_x)*x)
            y_full = int((self.o_y/self.f_y)*y)
            # noinspection PyTypeChecker
            temp_bgr = self.frame[y_full, x_full]
            temp_hsv = bgr2hsv(temp_bgr)
            print(temp_hsv)
            print(type(temp_bgr))
            self.relative_markers.append((x, y, temp_hsv))
            self.initial_markers.append((x_full, y_full, temp_hsv))
            cv2.circle(self.frame, (x_full, y_full), 2, (0, 0, 255), 2)
            cv2.imshow(self.name, cv2.resize(self.frame, (self.f_x, self.f_y)))

    def c_to_c_regularity(self, contour):
        (x, y), (w, h), _ = cv2.minAreaRect(contour)
        rect_c = (int(x+w/2), int(y+h/2))
        circ_c, _ = cv2.minEnclosingCircle(contour)
        mx, my, _ = self.current_relative_marker
        min_d = min(dist(rect_c, (mx, my)), dist(circ_c, (mx, my)))
        return self.prox_wt*min_d**2 + (1-self.prox_wt)*cv2.contourArea(contour)

    def show(self):
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        test_mask = cv2.inRange(self.hsv, (1, 1, 1), (179, 255, 255))
        test_mask[test_mask != 0] = 1
        canny = cv2.Canny(self.frame, 700, 751, apertureSize=5, L2gradient=True)
        cv2.imshow('color', cv2.resize(canny, (self.f_x, self.f_y)))
        canny[canny != 0] = 1
        #test = canny
        #test = np.ceil((test_mask + canny)/5)
        test2 = test_mask + canny
        print(test2)
        test2[test2 != 2] = 0
        print(test2)
        test2[test2 == 2] = 255
        print(test2)
        test2 = test2.astype(np.uint8)
        #cv2.imshow('mask', cv2.resize(test, (self.f_x, self.f_y)))
        cv2.imshow('mask', cv2.resize(test2, (self.f_x, self.f_y)))
        #cv2.imshow('color', cv2.resize(self.frame, (self.f_x, self.f_y)))
        #self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow(self.name, cv2.resize(cv2.Canny(self.frame, 750, 751, apertureSize=5, L2gradient=True), (self.f_x, self.f_y)))
        cv2.imshow('reg', cv2.resize(self.frame, (self.f_x, self.f_y)))
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                exit(5)

    def get_data(self):
        if self.d_out:
            data_df = pd.DataFrame(self.data[1:, :])
            data_df.columns = self.data_headers
            return data_df
        print('Cannot return data. WindowWrapper.d_out is disabled.')
        exit(0)

    def get_retrieval_state(self):
        return self.retv

    def get_name(self):
        return self.name

    def set_markers(self, mks):
        if len(mks) == len(self.markers):
            self.markers = mks
        else:
            print('Invalid marker array size')

    def get_markers(self):
        return self.markers

    def get_initial_markers(self):
        return self.initial_markers

    def set_act_frame(self, x, y):
        self.o_x = x
        self.o_y = y

    def get_act_frame(self):
        return self.o_x, self.o_y

    def set_frame(self, fr):
        self.frame = fr

    def get_frame(self):
        return self.frame

    def get_subframes(self):
        return self.subframes

    def set_tgt_dims(self, x, y):
        self.f_x = x
        self.f_y = y

    def get_tgt_dims(self):
        return self.f_x, self.f_y

    def next_frame(self):
        self.f_num += 1
        self.retv, self.frame = self.cap.read()
        self.oframe = copy.deepcopy(self.frame)
        return self.retv, self.oframe, self.frame

    def update_color_thresholds(self):
        markers_proxy = self.markers
        if len(self.markers) == 0:
            markers_proxy = self.initial_markers
        for l, marker in enumerate(markers_proxy):
            temp_thresh = copy.deepcopy(marker[2])
            h_noise = int(self.h_buf*180)
            s_noise = int(self.sv_buf*256)
            v_noise = s_noise
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
            self.thresh[l] = (bottom, top)
        #print(markers_proxy[0], self.thresh[0])

    def update_contours(self):
        # Convert BGR to HSV
        self.hsv = cv2.cvtColor(self.subframes[self.i], cv2.COLOR_BGR2HSV)
        #self.full_hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        # Threshold the HSV image to get only the chosen color
        # Mask contains a white on black image, where white pixels
        # represent that a value was within our threshold.
        if self._ac:
            self.update_color_thresholds()

        for m, thr in enumerate(self.thresh):
            self.color_masks[m] = cv2.inRange(self.hsv, thr[0], thr[1])

        # Find contours (distinct edges between two colors) in mask using OpenCV builtin
        # This function returns 2 values, but we only care about the first

        # Note: In some OpenCV versions this function will return 3 values, in which
        # case the second is the contours value. If you have one of those versions of
        # OpenCV, you will get an error about "unpacking values" from this line, which you
        # can fix by adding a throwaway variable before contours
            self.contours[m], _ = cv2.findContours(self.color_masks[m], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        '''
        print(self.lcb, self.ucb)
        cv2.imshow('auxframe', cv2.resize(self.color_mask, (1280, 720)))
        key = cv2.waitKey(1) & 0xFF
        while key != ord('q'):
            key = cv2.waitKey(1) & 0xFF
        exit(1)
        '''

    def set_initial_points(self, is_first_view, is_first_frame):
        if self._as:
            # Find the most circular contour by hull area
            cc_list = list(self.contours)
            cc_list.sort(reverse=True, key=regularity)
            self.contours = cc_list
            # contours_color = [max(contours_color, key=regularity)]
            max_significant = self.trackers
            #self.markers = [(0, 0, [])] * max_significant
            #self.relative_markers = [(0, 0, [])] * max_significant
            for j in range(max_significant):
                circle = self.contours[j]

                # Get a bounding rectangle around that contour
                # x, y, w, h = cv2.boundingRect(circle)
                c, r = cv2.minEnclosingCircle(circle)

                temp_color = bgr2hsv(self.subframes[self.i][int(c[0]), int(c[1])])
                self.markers[j] = (int(c[0]), int(c[1]), temp_color)
                self.relative_markers[j] = (int(c[0]), int(c[1]), temp_color)

                # Draw the rectangle on our frame
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 208, 255), 2)

                # Draw the circle on the frame
                if self.vis:
                    cv2.circle(self.frame, (self.markers[j][0], self.markers[j][1]), int(r), (0, 255, 0), 2)

                ''' Draws lines in the first frame, sometimes causes big skips
                for k, pt in enumerate(markers):
                    if k < len(markers)-1:
                        cv2.line(oframe, (markers[k][0], markers[k][1]), (markers[k+1][0], markers[k+1][1]), (0, 0, 255), 3)
                '''
        else:
            cv2.imshow(self.name, cv2.resize(self.frame, (self.f_x, self.f_y)))
            self.initial_markers = []
            while len(self.initial_markers) < self.trackers:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    exit(1)

        # is_first_frame set to false to bypass the single-frame view in order to assert marker values
        _, _, _ = self.subimage(is_first_view, False, set=True)
        for k, sf in enumerate(self.subframes):
            # Right here, markers are empty and thresholds are still at extremes
            #print('End of first pass initials', self.initial_markers)
            #print('End of first pass thresholds', self.thresh)
            self.update_contours()
            #print('Post-contour update initials', self.initial_markers)
            #print('Post-contour update thresholds', self.thresh)

            self.auto_update_markers(k, is_first_view, is_first_frame)
            #print('Post-auto-update markers', self.markers)
        self.markers = sorted(self.markers, key=marker_sort)
        self.initial_markers = copy.deepcopy(self.markers)
        return self.markers

    def subimage(self, is_first_view, is_first_frame, set=False):
        # Need to use oframes in the subimage method in order to avoid picking up the red marking dots in the subimages
        self.i = 0
        if is_first_frame:
            self.subframes = [self.oframe]
            return self.corners, self.subframes, self.tls

        self.corners = []
        self.subframes = []
        self.tls = []
        h = self.o_x
        w = self.o_y

        # Only runs when not first frame, but first view, which only happens during set_initial_points
        if set:
            markers_proxy = self.initial_markers
        else:
            markers_proxy = self.markers

        for center in markers_proxy:

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

    def auto_update_markers(self, i, is_first_view, is_first_frame):
        if is_first_view:
            '''
            print(self.initial_markers[i])
            print(self.thresh[i])
            if i == 2:
                cv2.imshow('mask', cv2.resize(self.color_masks[i], (self.f_x, self.f_y)))
                key = cv2.waitKey(1) & 0xFF
                while key != ord('q'):
                    key = cv2.waitKey(1) & 0xFF
                exit(0)
            '''
            self.current_relative_marker = self.initial_markers[i]
        else:
            self.current_relative_marker = self.relative_markers[i]

        if self._ac:
            p = i
        else:
            p = 0
        try:
            target_contour = max(self.contours[p], key=self.c_to_c_regularity)#regularity)#cv2.contourArea)

            c, r = cv2.minEnclosingCircle(target_contour)

            #Have to switch y and x to get the correct color
            y = int(c[1])# + 1
            x = int(c[0])# + 1
            temp_bgr = self.subframes[i][y, x]
            temp_hsv = bgr2hsv(temp_bgr)
            #print('Temp color', temp_bgr)

            self.relative_markers[i] = (x, y, temp_hsv)
            x_abs = self.tls[i][0] + self.relative_markers[i][0]
            y_abs = self.tls[i][1] + self.relative_markers[i][1]
            if is_first_view and is_first_frame:
                # noinspection PyTypeChecker
                self.markers.append((x_abs, y_abs, temp_hsv))  # self.frame[self.tls[i][0] + self.relative_markers[i][0], self.tls[i][1] + self.relative_markers[i][1]]))
            else:
                # noinspection PyTypeChecker
                self.markers[i] = (x_abs, y_abs, temp_hsv)  # self.frame[self.tls[i][0] + self.relative_markers[i][0], self.tls[i][1] + self.relative_markers[i][1]])

            if self.vis:
                # Draw the circle on the frame
                cv2.circle(self.frame, (self.markers[i][0], self.markers[i][1]), int(r), (0, 255, 0), 2)

                # Draw framing rectangles
                cv2.rectangle(self.frame, (self.corners[i][0][0], self.corners[i][0][1]),
                              (self.corners[i][1][0], self.corners[i][1][1]), (255, 0, 0), 2)
                cv2.circle(self.frame, (self.corners[i][0][0], self.corners[i][0][1]), 10, (255, 0, 0), 2)

                for k, pt in enumerate(self.markers):
                    if k < len(self.markers) - 1:
                        cv2.line(self.frame, (self.markers[k][0], self.markers[k][1]), (self.markers[k + 1][0], self.markers[k + 1][1]), (0, 0, 255), 5)

            temp_pt = (i+1, x_abs, y_abs, temp_hsv[0], temp_hsv[1], temp_hsv[2], 0)

        except ValueError as e:
            print(e)
            if is_first_view and is_first_frame:
                self.markers.append(self.initial_markers[i])
            temp_pt = (-1, 0, 0, 0, 0, 0, 1)

        if is_first_view and self.d_out:
            self.data = np.vstack((self.data, np.array(temp_pt)))

    def analyze_contours(self, is_first_view, is_first_frame):
        if len(self.contours) != 0:
            if is_first_view and is_first_frame:
                self.markers = self.set_initial_points(is_first_view, is_first_frame)
            elif is_first_frame:
                self.markers = self.initial_markers
            else:
                self.auto_update_markers(self.i, is_first_view, is_first_frame)
                self.i += 1

        if self.vis:
            if not self._as and not is_first_frame:
                cv2.imshow(self.name, cv2.resize(self.frame, (self.f_x, self.f_y)))
