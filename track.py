import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, track_id, bbox, frame_id):
        self.track_id = track_id
        self.bbox = bbox
        self.frame_id = frame_id
        self.tracked_positions = [bbox]
        self.average_speed = 0
        self.deleted = False

    def update(self, bbox, frame_id):
        self.tracked_positions.append(bbox)
        self.frame_id = frame_id
        self.bbox = bbox
        self.average_speed = self.average_speed + np.linalg.norm(bbox[:2] - self.tracked_positions[-2][:2])
        self.average_speed = self.average_speed / len(self.tracked_positions)

class HungarianTracker:
    def __init__(self, n_history: int = 50):
        self.detections = []
        self.predictions = []
        self.tracks = []
        self.n_history = n_history

    def get_assignments(self):
        """
            hungarian algorithm to match current and previous center points
            :param detections:
            :param predictions:
            :return:
            """
        # create cost matrix of current and previous center points which is square matrix
        shape = np.max([len(self.detections), len(self.predictions)])
        cost_matrix = np.full((shape, shape), 999999)

        detections, predictions = np.array(self.detections), np.array(self.predictions)
        for i, p in enumerate(predictions):
            for j, d in enumerate(detections):
                cost_matrix[i, j] = np.linalg.norm(d[:2] - p[:2])
        # solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return col_ind, cost_matrix[row_ind, col_ind]

    def get_last_bboxes(self, frame_id):
        """
            get last bboxes of all tracks up to n_history frames
            :param tracks:
            :return:
        """
        return [np.hstack([track.tracked_positions[-1], track.track_id]) for track in self.tracks
                if track.frame_id >= frame_id - self.n_history and not track.deleted]

    def get_tracks(self, detections, frame_id):
        """
            get tracks from detections and predictions
            :param frame_id:
            :param detections:
            :return:
            """
        self.detections = detections
        self.predictions = self.get_last_bboxes(frame_id)
        return self.make_tracks(frame_id)

    def make_tracks(self, frame_id):
        """
            make tracks from detections and predictions
            missing is when less detection than predictions. it can occur in two ways
            1. when a detection is lost
            2. when a detection is detected but not matched with a prediction
            unassociated is when more detections than predictions
            :param frame_id: current frame id
            :return:
            """
        # associate detections and predictions

        if len(self.tracks) <= 0:
            for i, d in enumerate(self.detections):
                self.tracks.append(Track(i, np.array([d[0], d[1], d[2], d[3]]), frame_id))  # track id, x, y, w, h
            return self.tracks
        assignments, costs = self.get_assignments()
        # add unassociated center points
        assigned = []
        for i, a in enumerate(assignments):
            if a < len(self.detections) and i < len(self.tracks) and costs[i] < 50:
                # if detection is associated, add it to track
                self.tracks[int(self.predictions[i][4])].update(self.detections[a], frame_id)
                assigned.append(a)
            else:
                continue
        # add missing center points
        for i, d in enumerate(self.detections):
            if i not in assigned:
                self.tracks.append(Track(len(self.tracks), d, frame_id))
        return self.tracks