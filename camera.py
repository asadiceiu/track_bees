import cv2
from webcamstream import WebcamVideoStream, FPS

from orb_detector import ORBTracker
from track import HungarianTracker


class VideoCap:
    def __init__(self, video_path=0, refresh_timeout: int = 500, is_direct: bool = False, height: int = 720, width: int = 1280):
        """
        Creates a VideoCap object with the given video path.
        :param video_path:
        :param refresh_timeout:
        """
        self.tracker = None
        self.fgbg = None
        self.refresh_timeout = refresh_timeout
        self.vs = cv2.VideoCapture(video_path, cv2.CAP_ANY) \
            if isinstance(video_path, int) else cv2.VideoCapture(video_path) # using cv2.CAP_DSHOW for directshow camera
        if not self.vs.isOpened():
            raise Exception("Could not open video")
        self.vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.fps = FPS().start()
        self.n_frames = 0
        self.current_frame = 0
        #self.total_frames = int(self.vs.get(cv2.CAP_PROP_FRAME_COUNT)) if video_path is not None else -1
        self.is_direct = is_direct
        self.height = height
        self.width = width


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
        # if 0 < self.total_frames <= self.current_frame:
        #     self.current_frame = 0
        #     self.vs.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, frame = self.vs.read()
        if not ret:
            raise Exception("couldn't grab image frame")
        self.fps.update()
        if self.is_direct:
            return frame
        ret, jpg = cv2.imencode(".jpg", frame)
        return jpg.tobytes()

    def draw_orb_tracks(self, frame, draw_kp: bool = True, draw_detections: bool = False, draw_tracks: bool = True, draw_numbers: bool = True):
        """
        Draws the ORB tracks on the given frame.
        :param frame:
        :return:
        """
        if self.tracker is None:
            return frame
        self.tracker.draw_tracks(frame, draw_kp, draw_detections, draw_tracks, draw_numbers)
        return frame

    def get_orb_tracking(self, draw_kp: bool = True, draw_detections: bool = False, draw_tracks: bool = True, draw_numbers: bool = True):
        """
        Returns the current frame with the ORB tracks drawn on it.
        :return:
        """
        ret, frame = self.vs.read()
        if not ret:
            raise Exception(f"couldn't grab image frame {self.current_frame}")
        self.current_frame += 1
        self.fps.update()
        if self.tracker is None:
            self.tracker = ORBTracker(heatmap_size=10)
        self.tracker.track(frame)
        frame = self.draw_orb_tracks(frame, draw_kp, draw_detections, draw_tracks, draw_numbers)
        self.write_bboxes_to_yolo_format()
        if self.is_direct:
            return frame
        ret, jpg = cv2.imencode(".jpg", frame)
        return jpg.tobytes()

    def write_bboxes_to_yolo_format(self):
        """
        Writes the bounding boxes to a file.
        :param bboxes:
        :return:
        """
        with open("bboxes.csv", "a+") as f:
            for bbox in self.tracker.get_detections():
                f.write(f"{self.current_frame}\t0\t{bbox[0]/self.width}\t{bbox[1]/self.height}\t{bbox[2]/self.width}\t{bbox[3]/self.width}\n")


    def get_background(self):
        """
        Returns the current background frame.
        :return:
        """

        self.current_frame += 1
        # if 0 < self.total_frames <= self.current_frame:
        #     self.current_frame = 0
        #     self.vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.vs.read()
        if not ret:
            raise Exception("couldn't grab image frame")
        self.fps.update()
        if self.fgbg is None:
            self.setup_background_subtraction()
        fgmask = self.fgbg.apply(frame)

        if self.is_direct:
            return frame
        ret, jpg = cv2.imencode(".jpg", fgmask)
        return jpg.tobytes()
