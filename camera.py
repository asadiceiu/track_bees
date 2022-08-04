import cv2
import numpy as np

from orb_detector import ORBTracker
from track import HungarianTracker


class VideoCap:
    def __init__(self, video_path=None, refresh_timeout: int = 500, is_direct: bool = False):
        """
        Creates a VideoCap object with the given video path.
        :param video_path:
        :param refresh_timeout:
        """
        self.tracker = None
        self.fgbg = None
        self.refresh_timeout = refresh_timeout
        self.vs = cv2.VideoCapture(video_path) if video_path is not None else cv2.VideoCapture(0)
        self.n_frames = 0
        self.current_frame = 0
        self.total_frames = int(self.vs.get(cv2.CAP_PROP_FRAME_COUNT)) if video_path is not None else -1
        self.is_direct = is_direct

    def setup_background_subtraction(self):
        """
        Sets up the background subtraction.
        :return:
        """
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        return

    def __del__(self):
        self.vs.release()

    def get_frame(self):
        """
        Returns the current frame without any modification to it.
        :return:
        """
        self.current_frame += 1
        if 0 < self.total_frames <= self.current_frame:
            self.current_frame = 0
            self.vs.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, frame = self.vs.read()
        if not ret:
            raise Exception("couldn't grab image frame")

        if self.is_direct:
            return frame
        ret, jpg = cv2.imencode(".jpg", frame)
        return jpg.tobytes()

    def draw_orb_tracks(self, frame):
        """
        Draws the ORB tracks on the given frame.
        :param frame:
        :return:
        """
        if self.tracker is None:
            return frame
        self.tracker.draw_tracks(frame, draw_kp=True, draw_detections=False)
        return frame

    def get_orb_tracking(self):
        """
        Returns the current frame with the ORB tracks drawn on it.
        :return:
        """
        ret, frame = self.vs.read()
        if not ret:
            raise Exception("couldn't grab image frame")
        if self.tracker is None:
            self.tracker = ORBTracker()
        self.tracker.track(frame)
        frame = self.draw_orb_tracks(frame)

        if self.is_direct:
            return frame
        ret, jpg = cv2.imencode(".jpg", frame)
        return jpg.tobytes()

    def get_background(self):
        """
        Returns the current background frame.
        :return:
        """

        self.current_frame += 1
        if 0 < self.total_frames <= self.current_frame:
            self.current_frame = 0
            self.vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.vs.read()
        if not ret:
            raise Exception("couldn't grab image frame")
        if self.fgbg is None:
            self.setup_background_subtraction()
        fgmask = self.fgbg.apply(frame)

        if self.is_direct:
            return frame
        ret, jpg = cv2.imencode(".jpg", fgmask)
        return jpg.tobytes()
