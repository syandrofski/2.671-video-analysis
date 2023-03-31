import sys
import cv2
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib as plt
from math import sin, cos, tan, atan, sqrt
import time
import copy
from imutils.video import VideoStream
from imutils.video import FPS
import argparse


def main():
    tracker = cv2.legacy_TrackerKCF_create
    cap = cv2.VideoCapture('C:\\Users\\spenc\\Dropbox (MIT)\\2.671 Go Forth and Measure\\jump1_AdobeExpress.mp4')
    retv = True
    while True:
        retv, frame = cap.read()
        if not retv:
            cap = cv2.VideoCapture(
                'C:\\Users\\spenc\\Dropbox (MIT)\\2.671 Go Forth and Measure\\jump1_AdobeExpress.mp4')
        if retv:
            cv2.imshow('frame', cv2.resize(frame, (1280, 720)))
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            exit(0)


if __name__ == '__main__':
    main()
