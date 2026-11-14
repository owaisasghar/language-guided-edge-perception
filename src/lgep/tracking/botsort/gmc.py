# Vendored from https://github.com/NirAharon/BoT-SORT/blob/251985436d6712aaf682aaaf5f71edb4987224bd/tracker/gmc.py
# (MIT licence, see LICENSE in this folder). Changes are marked "lgep:":
#   1. kept the sparse optical flow method (BoT-SORT's default for live video; OpenCV only) and "none".
#      Dropped: ORB / SIFT / ECC and "file" (pre-computed VidStab warps for the MOTChallenge sequences,
#      which cannot exist for a live camera) and the matplotlib debug drawing.
#   2. two cases in which the official code raises and a live camera must not stop: a frame without corners
#      (goodFeaturesToTrack returns None) and a failed RANSAC fit (estimateAffinePartial2D returns None)
#      both give the identity warp. The warning print became a debug log.
#   3. ``rgb=True`` converts an RGB frame to grey (the official code assumes BGR).
# ruff: noqa

import copy
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


class GMC:
    """Global motion compensation: the 2 x 3 similarity warp of the camera image from one frame to the next."""

    def __init__(self, method='sparseOptFlow', downscale=2, verbose=None):
        super(GMC, self).__init__()

        self.method = method
        self.downscale = max(1, int(downscale))

        if self.method == 'sparseOptFlow':
            self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3,
                                       useHarrisDetector=False, k=0.04)

        elif self.method == 'none' or self.method == 'None':
            self.method = 'none'
        else:
            raise ValueError("Error: Unknown CMC method:" + method)  # lgep: orb / sift / ecc / file dropped

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None

        self.initializedFirstFrame = False

    def apply(self, raw_frame, detections=None, rgb=False):  # lgep: + rgb
        if self.method == 'sparseOptFlow':
            return self.applySparseOptFlow(raw_frame, detections, rgb)
        else:
            return np.eye(2, 3)

    def applySparseOptFlow(self, raw_frame, detections=None, rgb=False):  # lgep: + rgb

        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY)  # lgep: + RGB
        H = np.eye(2, 3)

        # Downscale image
        if self.downscale > 1.0:
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # find the keypoints
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        if self.prevKeyPoints is None or len(self.prevKeyPoints) == 0:  # lgep: e.g. a black frame
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            return H

        # find correspondences
        matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)

        # leave good correspondences only
        prevPoints = []
        currPoints = []

        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            if H is None:  # lgep: RANSAC found no model
                H = np.eye(2, 3)
            # Handle downscale
            elif self.downscale > 1.0:  # lgep: was if
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            log.debug('not enough matching points for camera motion')  # lgep: was print('Warning: ...')

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        return H
