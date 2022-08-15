import cv2
import numpy as np

from track import HungarianTracker


class ORBDetector:
    def __init__(self, prune_bg: bool = True, refresh_bg_frame: int = 10, heatmap_size: int = 10,
                 min_hits: int = 3, min_detection_area: int = 50, max_detection_area: int = 5000):
        """
        Creates a ORBDetector object. prune_bg determines whether to prune the background keypoints and descriptors.
        refresh_bg_frame determines how often to refresh the background keypoints and descriptors.
        heatmap_size determines the size of the heatmap. heatmap_threshold determines the threshold for the heatmap.
        :param prune_bg: Whether to prune the background keypoints and descriptors.
        :param refresh_bg_frame: the number of frames after which to refresh the background keypoints and descriptors.
        :param heatmap_size: size of the heatmap
        :param min_hits: minimum number of hits to be considered a detection.
        """
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.bgsub = cv2.createBackgroundSubtractorMOG2()
        self.kp_bg = None
        self.des_bg = None
        self.kp_frame = None
        self.des_frame = None
        self.prune_bg = prune_bg
        self.n_frames = 0
        self.refresh_bg_frames = refresh_bg_frame
        self.heatmap_size = heatmap_size
        self.min_hits = min_hits
        self.min_detection_area = min_detection_area
        self.max_detection_area = max_detection_area

    def get_frame_keypoints(self):
        """
        Gets the keypoints of the frame.
        :return:
        """
        return self.kp_frame

    def get_frame_descriptors(self):
        """
        Gets the descriptors of the frame.
        :return:
        """
        return self.des_frame

    def get_bg_descriptors(self):
        """
        Gets the background descriptors.
        :return:
        """
        return self.des_bg

    def get_bg_keypoints(self):
        """
        Gets the background keypoints.
        :return:
        """
        return self.kp_bg

    def _compute_bg(self, frame):
        """
        Computes the background keypoints and descriptors.
        :param frame:
        :return:
        """
        self.bgsub.apply(frame)
        if self.n_frames % self.refresh_bg_frames == 0 or self.n_frames == 1:
            self.kp_bg, self.des_bg = self.orb.detectAndCompute(self.bgsub.getBackgroundImage(), None)
        return

    def _prune_matches(self):
        """
        Prunes the matches between the background and the frame keypoints.
        :return:
        """
        matches = self.bf.match(self.des_frame, self.des_bg)
        match_idx = np.array([m.queryIdx for m in matches])
        bg_idx = np.array([i for i in range(len(self.kp_frame)) if i not in match_idx])
        self.kp_frame = np.array([self.kp_frame[i] for i in bg_idx])
        self.des_frame = np.array([self.des_frame[i] for i in bg_idx])
        return self.kp_frame, self.des_frame

    def _get_keypoints(self, image):
        """
        Gets the keypoints and descriptors of the frame.
        :param image:
        :return:
        """
        self.n_frames += 1
        self.kp_frame, self.des_frame = self.orb.detectAndCompute(image, None)
        if self.prune_bg:
            self._compute_bg(image)
            self.kp_frame, self.des_frame = self._prune_matches()
        return self.kp_frame, self.des_frame

    def get_detections(self, image):
        """
        Gets the detections of the frame.
        :param image:
        :return:
        """
        self._get_keypoints(image)
        heatmap = self._create_heatmap(image)
        contours, _ = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            # write area to heatmap
            #cv2.putText(heatmap, str(area), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            #print(area, self.min_detection_area, self.max_detection_area, end="\r\n")
            if self.max_detection_area > area > self.min_detection_area:
                cv2.drawContours(heatmap, [contour], -1, (255, 255, 255), 2)
                detections.append(np.array([x + w // 2, y + h // 2, w, h])) # x, y, w, h

        return detections

    def _create_heatmap(self, image):
        """
        Creates the heatmap.
        :param image:
        :return:
        """
        heatmap = np.zeros(image.shape[:2], np.uint8)
        for kp in self.kp_frame:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            heatmap[y - self.heatmap_size:y + self.heatmap_size, x - self.heatmap_size:x + self.heatmap_size] += 1
        heatmap[heatmap < self.min_hits] = 0
        return heatmap


class ORBTracker:
    def __init__(self, prune_bg: bool = True, refresh_bg_frame: int = 10, heatmap_size: int = 20,
                 min_hits: int = 3, min_track_length: int = 5, min_detection_area: int = 100, max_detection_area: int = 5000, refresh_frame_count: int = 1000):
        """
        Creates a ORBTracker object. prune_bg determines whether to prune the background keypoints and descriptors.
        refresh_bg_frame determines how often to refresh the background keypoints and descriptors.
        heatmap_size determines the size of the heatmap. heatmap_threshold determines the threshold for the heatmap.
        min_track_length determines the minimum length of a track.

        :param prune_bg:
        :param refresh_bg_frame:
        :param heatmap_size:
        :param min_track_length:
        """
        self.detections = []
        self.prune_bg = prune_bg
        self.refresh_bg_frame = refresh_bg_frame
        self.heatmap_size = heatmap_size
        self.min_hits = min_hits
        self.min_track_length = min_track_length
        self.min_detection_area = min_detection_area
        self.max_detection_area = max_detection_area
        self.refresh_frame_count = refresh_frame_count
        self.n_frames = 0
        self.orb_detector = ORBDetector(prune_bg, refresh_bg_frame, heatmap_size, min_hits=min_hits,
                                        min_detection_area=min_detection_area, max_detection_area=max_detection_area)
        self.tracker = HungarianTracker(n_history=50)

    def draw_tracks(self, image, draw_kp: bool = True, draw_detections: bool = True, draw_tracks: bool = True,
                    draw_numbers: bool = True):
        """
        Draws the tracks on the image.
        :param image:
        :return:
        """
        if draw_kp:
            cv2.drawKeypoints(image, self.orb_detector.get_frame_keypoints(), image, color=(0, 255, 0))

        if draw_detections:
            for detection in self.detections:
                x, y, w, h = detection
                cv2.rectangle(image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 1)

        for track in self.get_tracks():
            if len(track.tracked_positions) < self.min_track_length:
                continue
            x, y, w, h = track.tracked_positions[-1].astype(int)
            if draw_numbers:
                cv2.putText(image, str(track.track_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            cv2.circle(image, (x, y), 2, (0, 0, 220), -1)
            if draw_tracks:
                l = len(track.tracked_positions)
                for i in range(l - 1):
                    x1, y1, _, _ = track.tracked_positions[i].astype(int)
                    x2, y2, _, _ = track.tracked_positions[i + 1].astype(int)
                    cv2.line(image, (x1, y1), (x2, y2), (0, 120, 255 - (l - i) * 3), 1)

        return image

    def track(self, image):
        """
        Tracks the detections.
        :param image:
        :return:
        """
        self.n_frames += 1
        if self.orb_detector.n_frames % self.refresh_frame_count == 0:
            self.orb_detector = ORBDetector(self.prune_bg, self.refresh_bg_frame, self.heatmap_size,
                                            min_hits=self.min_hits,
                                            min_detection_area=self.min_detection_area,
                                            max_detection_area=self.max_detection_area)
            self.tracker = HungarianTracker(n_history=50)
            self.n_frames = 0
        print("Tracking frame {}".format(self.n_frames), end="\r")
        self.detections = self.orb_detector.get_detections(image)
        self.tracker.get_tracks(self.detections, self.n_frames)
        return self.tracker.tracks

    def get_tracks(self):
        """
        Gets the tracks.
        :return:
        """
        return self.tracker.tracks

    def get_detections(self):
        """
        Gets the detections.
        :return:
        """
        return self.detections
