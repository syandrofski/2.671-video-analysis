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


def hsv_color_inverse(color):
    """Returns the inverse of an HSV color code."""
    # Get maximum values for H, S, and V channels
    max_hue = 180
    max_sat = 255
    max_val = 255

    # Invert the hue
    inverted_hue = (color[0] + int(max_hue/2)) % max_hue

    # Invert the saturation and value
    inverted_sat = max_sat - color[1]
    inverted_val = max_val - color[2]

    # Return the inverted color
    return np.array([inverted_hue, inverted_sat, inverted_val])


def check():
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        exit(0)
    else:
        return key


class WindowWrapper:

    def __init__(self, n, targets=3, rsz_factor=0.5, fpath='C:\\Users\\spenc\\Dropbox (MIT)\\2.671 Go Forth and Measure\\test.mp4',
                 marker_buffer=0.025, hue_buffer=0.025, sat_buffer=0.25, val_buffer=0.25, visualize=True,
                 area_weight=0.2, color_weight=0.2, distance_weight=0.2, circularity_weight=0.2, filled_weight=0.2,
                 hyper=True, canny_thresh1=700, canny_thresh2=751, canny_apertureSize=5, canny_L2threshold=True,
                 error_threshold=0.5, debug=False):
        self.path = fpath
        self.name = n
        self.vis = visualize
        self.parsing = True
        self.debug = debug
        self.augment_canny = hyper
        self.err_thresh = error_threshold

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
        self.compare_tracker = 0
        self.temp = (np.array([]), 0)

        self.contours = [[]] * self.trackers
        self.subframes = []
        self.sf_hsv = []
        self.sf_canny = []
        self.hyper = []
        self.sf_mask = []
        self.rsz = rsz_factor

        self.h_buf = hue_buffer
        self.s_buf = sat_buffer
        self.v_buf = val_buffer

        self.area_wt = area_weight
        self.color_sim_wt = color_weight
        self.point_dist_wt = distance_weight
        self.circ_wt = circularity_weight
        self.fill_wt = filled_weight

        self.canny_lower = canny_thresh1
        self.canny_upper = canny_thresh2
        self.canny_ap_size = canny_apertureSize
        self.canny_L2 = canny_L2threshold

        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self.on_click)

        # cap is a VideoCapture object we can use to get frames from video files
        self.cap = cv2.VideoCapture(self.path)

        # Gets an initial frame
        self.f_num = 0
        self.next_frame()

        # Resets capture file
        self.cap = cv2.VideoCapture(self.path)
        self.f_num = 0
        self.sf_num = 0

        self.o_x = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.o_y = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.f_x = int(self.rsz * self.o_x)
        self.f_y = int(self.rsz * self.o_y)

        self.thresh = (np.array([0, 0, 0]), np.array([179, 255, 255]))
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
            temp_bgr = self.oframe[y_full, x_full]
            temp_hsv = bgr2hsv(temp_bgr)

            self.current[self.x, self.selections] = x_full
            self.current[self.y, self.selections] = y_full
            self.current[self.rx, self.selections] = x
            self.current[self.ry, self.selections] = y
            self.current[self.px, self.selections] = x_full
            self.current[self.py, self.selections] = y_full
            self.current[self.h, self.selections] = temp_hsv[0]
            self.current[self.s, self.selections] = temp_hsv[1]
            self.current[self.v, self.selections] = temp_hsv[2]
            self.current[self.pt, self.selections] = 0
            self.current[self.conf, self.selections] = 1
            self.current[self.err, self.selections] = 0
            self.selections += 1

            inv_hsv = hsv_color_inverse(temp_hsv)
            inv_hsv = (int(inv_hsv[0]), int(inv_hsv[1]), int(inv_hsv[2]))

            cv2.circle(self.frame, (x_full, y_full), 2, inv_hsv, 2)
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
        if self.debug:
            print('\n\n')
        self.retv, self.oframe = self.cap.read()
        if self.debug:
            print(self.f_num)
        if self.retv:
            self.f_num += 1

            for t in range(self.trackers):
                if self.current[self.err, t] == 0:
                    self.last[:, t] = self.current[:, t]
                else:
                    self.last[self.err, t] = 1
                    self.last[self.pt, t] = self.current[self.pt, t]
                    self.last[self.buf, t] = self.current[self.buf, t]
                if self.debug:
                    print('LAST FRAME:\n', self.current[:, t].flatten())

            #self.last = np.copy(self.current)
            self.adv_struct = np.vstack((self.adv_struct, np.reshape(self.current, (1, self.current.shape[0], self.current.shape[1]))))
            self.current = np.zeros((len(self.data_headers), self.trackers))

        self.frame = copy.deepcopy(self.oframe)
        if self.retv:
            self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            self.canny = cv2.Canny(cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY), self.canny_lower, self.canny_upper, apertureSize=self.canny_ap_size, L2gradient=self.canny_L2)

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

    def subimage(self):
        # Need to use oframes in the subimage method in order to avoid picking up the red marking dots in the subimages
        self.sf_num = 0

        self.subframes = []
        self.sf_hsv = []
        self.sf_canny = []
        self.sf_mask = []
        self.hyper = []

        for i in range(self.trackers):
            if self.last[self.err, i] == 1:
                self.current[self.buf, i] = int(self.last[self.buf, i] + 0.1 * self.m_buf)
                if self.current[self.buf, i] >= self.o_y/2:
                    self.current[self.buf, i] = int(self.o_y/2.1)
                if self.current[self.buf, i] >= self.o_x/2:
                    self.current[self.buf, i] = int(self.o_x/2.1)
            else:
                self.current[self.buf, i] = self.m_buf

            self.current[self.tlx, i] = int(self.current[self.px, i] - self.current[self.buf, i])
            self.current[self.tly, i] = int(self.current[self.py, i] - self.current[self.buf, i])

            xmin = int(self.current[self.tlx, i])
            xmax = int(self.current[self.tlx, i] + int(2 * self.current[self.buf, i]))
            ymin = int(self.current[self.tly, i])
            ymax = int(self.current[self.tly, i] + int(2 * self.current[self.buf, i]))

            # For some reason for slicing, X and Y are switched and it's stupid
            self.subframes.append(self.oframe[ymin:ymax, xmin:xmax])

            temp_hsv = self.hsv[ymin:ymax, xmin:xmax]
            self.sf_hsv.append(temp_hsv)

            temp_canny = self.canny[ymin:ymax, xmin:xmax]
            temp_canny[temp_canny != 0] = 255
            self.sf_canny.append(temp_canny)


            self.thresh = self.update_color_threshold(i)
            if self.debug:
                print('THRESHOLDS:\n', self.thresh)

            temp_range = cv2.inRange(temp_hsv, self.thresh[0], self.thresh[1])
            self.sf_mask.append(temp_range)
            temp_range[temp_range != 0] = 255
            #cv2.imshow('test_frame', temp_range)

            if self.augment_canny:
                temp_hyper = temp_canny + temp_range
            else:
                temp_hyper = np.copy(temp_canny)
            temp_hyper[temp_hyper != 0] = 255
            temp_hyper = temp_hyper.astype(np.uint8)
            self.hyper.append(temp_hyper)

            '''
            #cv2.imshow('test_frame', self.hyper[i])
            key = cv2.waitKey(1) & 0xFF
            while key != ord('n') and key != ord('q'):
                key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                exit(201)
            '''

    def update_color_threshold(self, i):
        temp_thresh = (self.last[self.h, i], self.last[self.s, i], self.last[self.v, i])
        #if self.f_num > 10:
        #    temp_thresh = np.mean(self.adv_struct[-10:, self.h:self.v+1, i], axis=0).flatten()
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
        if err_codes[-1] == 0:# and err_codes[-2] == 0:
            return 2 #------------------------------------------- Switched from 1 to 2, need to patch the 1 for post-error and revert eventually
        if np.sum(err_codes) == len(err_codes) - 1:
            return 4
        if np.sum(err_codes)/len(err_codes) < 0.5:
            return 2
        return 3

    def pre_subimage_project(self):
        for i in range(self.trackers):
            self.current[self.pt, i] = self.assert_projection_type(i)

            self.calculate_projections(i)

    def calculate_projections(self, i):
        if self.adv_struct.shape[0] != 1:
            self.current[self.px, i] = self.last[self.x, i]
            self.current[self.py, i] = self.last[self.y, i]
        #return True

        pt = self.current[self.pt, i]

        if pt == 0 or pt == 1:
            difference_x = np.diff(self.adv_struct[-2:, self.x, i].flatten())/2
            difference_y = np.diff(self.adv_struct[-2:, self.y, i].flatten())/2

            self.current[self.px, i] = self.last[self.x, i] + difference_x
            self.current[self.py, i] = self.last[self.y, i] + difference_y

            self.current[self.buf, i]  = self.m_buf
        elif pt == 2:
            err_codes = self.adv_struct[:, self.err, i].flatten()
            healthy = np.array(np.where(err_codes == 0)).flatten()
            last = healthy[-1]
            stl = healthy[-2]
            span = abs(stl-last)

            difference_rate_x = (self.adv_struct[last, self.x, i] - self.adv_struct[stl, self.x, i])/span
            difference_rate_y = (self.adv_struct[last, self.y, i] - self.adv_struct[stl, self.y, i])/span

            proj_x = difference_rate_x * (self.f_num - 1 - last)
            proj_y = difference_rate_y * (self.f_num - 1 - last)

            self.current[self.px, i] = self.last[self.x, i] + proj_x
            self.current[self.py, i] = self.last[self.y, i] + proj_y

            #self.current[self.buf, i] = self.m_buf * (1 + last/10.0)
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

    def contour_compare(self, contour, verbose=False):
        self.compare_tracker += 1
        c, r = cv2. minEnclosingCircle(contour)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if r != 0 and hull_area != 0:
            circularity = 3.14*r**2/hull_area
        else:
            circularity = 0
        if circularity > 1:
            circularity = 1/circularity

        mar = cv2.minAreaRect(contour)
        rc, (w, h), a = mar
        rect_area = w*h

        last_area = self.last[self.hull, self.i_tracker]

        if not rect_area == 0:
            area_ratio = rect_area/last_area
            if area_ratio > 1:
                area_ratio = 1 / area_ratio
        else:
            return 0

        mini_tlx = int(c[0] - r)
        mini_tly = int(c[1] - r)
        mini_brx = int(c[0] + r)
        mini_bry = int(c[1] + r)
        corners = np.array([mini_tlx, mini_tly, mini_brx, mini_bry])
        corners[corners < 0] = 0
        corners[corners > 2 * self.current[self.buf, self.i_tracker]] = 2 * self.current[self.buf, self.i_tracker] - 1
        mini_tlx = int(corners[0])
        mini_tly = int(corners[1])
        mini_brx = int(corners[2])
        mini_bry = int(corners[3])
        mini = self.hyper[self.i_tracker][mini_tly:mini_bry+1, mini_tlx:mini_brx+1] #----------------Potentially change to match selected analysis frame, i.e. sf_mask---------
        pct_fill = np.count_nonzero(mini) / float(mini.size)

        '''
        sft = np.copy(self.subframes[0])
        cv2.drawContours(sft, [contour, np.int0(cv2.boxPoints(mar))], -1, (0, 0, 255), 2)
        cv2.circle(sft, (int(c[0]), int(c[1])), int(r), (255, 0, 0), 2)
        cv2.imshow('test_frame', cv2.resize(sft, (250, 250)))
        key = cv2.waitKey(1) & 0xFF
        while key != ord('n') and key != ord('q'):
            key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            exit(301)
        '''

        # Again, reverse y and x to pull hsv from a numpy array
        hsv = self.sf_hsv[self.i_tracker][int(c[1]), int(c[0])]
        last_hsv = np.array([self.last[self.h, self.i_tracker], self.last[self.s, self.i_tracker], self.last[self.s, self.i_tracker]])
        color_dist = 1 - hsv_distance(hsv, last_hsv)
        point_dist = dist(c[0], c[1], self.current[self.px, self.i_tracker]-self.current[self.tlx, self.i_tracker], self.current[self.py, self.i_tracker]-self.current[self.tly, self.i_tracker])
        point_dist = point_dist / self.current[self.buf, self.i_tracker]

        if point_dist > 1:
            point_dist = 1.0
        point_dist = 1 - point_dist

        if verbose and self.debug:
            print('components;', area_ratio, color_dist, point_dist, circularity, pct_fill)

        confidence = self.area_wt * area_ratio + self.color_sim_wt * color_dist + self.point_dist_wt * point_dist + self.circ_wt * circularity + self.fill_wt * pct_fill

        if verbose and self.debug:
            print('confidence;', confidence)

        return confidence

    def analyze_subframe(self, i):
        self.compare_tracker = 0
        self.contours[i], _ = cv2.findContours(self.sf_mask[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #formerly hyper instead of sf_mask

        self.i_tracker = i

        if len(self.contours[i]) < 1:
            if self.debug:
                print('No contours found!')
            self.current[self.conf, i] = 0
            self.temp = np.array([])
            x, y, w, h, a = (-1,-1, -1, -1, -1)
            self.temp = (np.array([]), i)
        else:
            target_contour = max(self.contours[i], key=cv2.contourArea)#self.contour_compare)#cv2.contourArea)

            self.temp = (target_contour, i)

            self.current[self.conf, i] = self.contour_compare(target_contour, verbose=True)
            (x, y), (w, h), a = cv2.minAreaRect(target_contour)

        if self.debug:
            print('last; ', self.last[self.conf, i])
            print('current; ', self.current[self.conf, i])

        if self.current[self.conf, i] < self.err_thresh*self.last[self.conf, i]:
            self.current[self.err, i] = 1
            if self.debug:
                print(self.f_num, self.current[self.conf, i])
            self.current[self.conf, i] = self.last[self.conf, i]
        else:
            self.current[self.hull, i] = w*h
            self.current[self.rx, i] = x
            self.current[self.ry, i] = y
            self.current[self.x, i] = x + self.current[self.tlx, i]
            self.current[self.y, i] = y + self.current[self.tly, i]

            # Switching y and x once again
            # noinspection PyTypeChecker
            temp_bgr = self.oframe[int(self.current[self.y, i]), int(self.current[self.x, i])]
            temp_hsv = bgr2hsv(temp_bgr)

            self.current[self.h, i] = temp_hsv[0]
            self.current[self.s, i] = temp_hsv[1]
            self.current[self.v, i] = temp_hsv[2]

            #print('error code', self.current[self.err, i])

            self.i_tracker = -1

    def draw(self):
        if self.vis:
            for i in range(self.trackers):
                if self.debug:
                    print('CURRENT FRAME:\n', self.current[:, i])
                m_cent = (int(self.current[self.x, i]), int(self.current[self.y, i]))
                b = int(self.current[self.buf, i])
                tl = (int(self.current[self.tlx, i]), int(self.current[self.tly, i]))
                br = (int(self.current[self.tlx, i] + 2 * b), int(self.current[self.tly, i] + 2 * b))

                if self.trackers > 1 and i < (self.trackers - 1) and self.current[self.err, i] + self.current[self.err, i+1] == 0:
                    cv2.line(self.frame, m_cent,(int(self.current[self.x, i+1]), int(self.current[self.y, i+1])), (255, 255, 0), 2)

                if self.current[self.err, i] == 1:
                    #marker_color = (0, 0, 255)
                    box_color = (0, 0, 255)
                else:
                    green = self.current[self.conf, i] * 255
                    red = 255 - green
                    marker_color = (0, green, red)
                    box_color = (150, 150, 150)
                    cv2.circle(self.frame, m_cent, 2, marker_color, 2)


                cv2.rectangle(self.frame, tl, br, box_color, 1)
                if self.debug:
                    print(tl, br)
                cv2.imshow(self.name, cv2.resize(self.frame, (self.f_x, self.f_y)))

            if self.debug:
                cv2.imshow('canny', cv2.resize(self.canny, (self.f_x, self.f_y)))
                #tmp = np.copy(self.hyper[self.trackers-1])
                tmp = np.copy(self.sf_mask[self.trackers-1])
                tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
                sframe, si = self.temp
                if self.current[self.err, si] == 0:
                    cv2.drawContours(tmp, [sframe], -1, (0, 0, 255), 1)
                cv2.imshow('hyper', cv2.resize(tmp, (250, 250)))

                key = check()
                while key != ord('n'):
                    key = check()

            key = check()

    def get_first_points(self):
        self.f_num = 0
        cv2.imshow(self.name, cv2.resize(self.frame, (self.f_x, self.f_y)))

        while self.selections < self.trackers:
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit(1)

        #self.current = np.transpose(self.current)
        #self.current = self.current[self.current[:, self.y].argsort()]
        #self.current = np.transpose(self.current)

        self.last = np.copy(self.current)
        self.adv_struct = np.reshape(self.current, (1, self.current.shape[0], self.current.shape[1]))
        self.subimage()
        self.get_hulls_initial()
        self.last = np.copy(self.current)
        self.adv_struct = np.reshape(self.current, (1, self.current.shape[0], self.current.shape[1]))
        self.analyze_all_subframes()

    def analyze_all_subframes(self):
        for i in range(self.trackers):
            self.analyze_subframe(i)

    def get_hulls_initial(self):
        for i in range(self.trackers):
            self.contours[i], _ = cv2.findContours(self.sf_mask[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #print(len(self.contours[i]))
            if len(self.contours[i]) == 0:
                print('Point ' + str(i + 1) + ' did not contain a representative marker. Please reselect.')
                exit(101)
            target_contour = max(self.contours[i], key=cv2.contourArea)

            if self.debug:
                #print sf_mask and contour
                sfm = np.copy(self.sf_mask[i])
                cv2.drawContours(sfm, [target_contour], -1, (0, 0, 255), 2)
                cv2.imshow('subframe mask', sfm)
                key = cv2.waitKey(1) & 0xFF
                while key != ord('q') and key != ord('n'):
                    key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    exit(400)
            '''
            cv2.imshow('test_frame', self.hyper[i])
            #print(len(self.contours[i]))
            print('at hypers2')
            print(self.hyper[i])
            key = cv2.waitKey(1) & 0xFF
            while key != ord('q'):
                key = cv2.waitKey(1) & 0xFF
            exit(102)
            '''

            _, (w, h), _ = cv2.minAreaRect(target_contour)

            self.current[self.hull, i] = w * h

    def replay(self):
        while True:
            self.cap = cv2.VideoCapture(self.path)
            self.f_num = 1
            self.retv = True
            while self.retv and self.f_num < self.adv_struct.shape[0]:
                self.nf_replay()
                key = check()
                if key == ord('t'):
                    key = ord('a')
                    time.sleep(0.01)
                    while key != ord('t') and key != ord('p'):
                        key = check()
            key = check()
            while key != ord('p'):
                key = check()

    def nf_replay(self):
        self.retv, self.frame = self.cap.read()
        self.current = self.adv_struct[self.f_num]
        self.draw()
        self.f_num += 1

    def export_data(self):
        return self.adv_struct

    def set_data(self, struc):
        self.adv_struct = struc
        self.trackers = self.adv_struct.shape[2]

    def interpolate(self):
        ct = 0
        rec = 0
        for i in range(self.trackers):
            ct = 0
            rec = 0
            for j in range(self.adv_struct.shape[0]):
                if self.adv_struct[j, self.err, i] and ct == 0:
                    ct += 1
                    rec = i - 1
                elif ct > 0:
                    x_rate = (self.adv_struct[j, self.x, i] - self.adv_struct[rec, self.x, i])/float(j - rec)
                    y_rate = (self.adv_struct[j, self.y, i] - self.adv_struct[rec, self.y, i])/float(j - rec)
                    xb = self.adv_struct[rec, self.x, i]
                    yb = self.adv_struct[rec, self.y, i]
                    tlxb = self.adv_struct[rec, self.tlx, i]
                    tlyb = self.adv_struct[rec, self.tly, i]
                    for k in range(rec + 1, j):
                        self.adv_struct[k, self.x, i] = xb + (k-rec)*x_rate
                        self.adv_struct[k, self.y, i] = yb + (k-rec)*y_rate
                        self.adv_struct[k, self.tlx, i] = tlxb+ (k-rec)*x_rate
                        self.adv_struct[k, self.tly, i] = tlyb+ (k-rec)*y_rate
                        self.adv_struct[k, self.buf, i] = self.m_buf
                    ct = 0
                    rec = 0
