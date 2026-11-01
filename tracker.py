import math
import time
from typing import List, Tuple

import numpy as np
try:
    from filterpy.kalman import KalmanFilter
    HAVE_FILTERPY = True
except Exception:
    KalmanFilter = None
    HAVE_FILTERPY = False
try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except Exception:
    linear_sum_assignment = None
    HAVE_SCIPY = False


class Track:
    def __init__(self, tid: int, bbox: Tuple[float, float, float, float], label: str, conf: float, timestamp: float, max_age: float = 18.0):
        # bbox = (x1, y1, x2, y2)
        self.id = tid
        self.bbox = bbox
        self.label = label
        self.last_seen = timestamp
        self.created_at = timestamp
        self.hits = 1
        self.age = 0
        self.counted = False
        self.last_conf = float(conf)
        self.max_age = max_age

        # Tracking state: prefer Kalman if available, otherwise fall back to simple centroid velocity
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        if HAVE_FILTERPY and KalmanFilter is not None:
            kf = KalmanFilter(dim_x=4, dim_z=2)
            dt = 1.0
            kf.F = np.array([[1, 0, dt, 0],
                             [0, 1, 0, dt],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=float)
            kf.H = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]], dtype=float)
            kf.P *= 1000.0
            kf.R *= 5.0
            kf.Q = np.eye(4) * 0.01
            kf.x = np.zeros((4, 1), dtype=float)
            kf.x[0, 0] = cx
            kf.x[1, 0] = cy
            self.kf = kf
            self._use_kalman = True
        else:
            # fallback: simple centroid-based velocity estimate
            self.prev_cx = cx
            self.prev_cy = cy
            self.vx = 0.0
            self.vy = 0.0
            self._use_kalman = False

    def predict(self, dt: float = 1.0) -> Tuple[float, float]:
        if self._use_kalman:
            self.kf.F[0, 2] = dt
            self.kf.F[1, 3] = dt
            self.kf.predict()
            cx = float(self.kf.x[0, 0])
            cy = float(self.kf.x[1, 0])
            return cx, cy
        else:
            # return last known centroid (no prediction)
            return float(self.prev_cx), float(self.prev_cy)

    def update(self, bbox: Tuple[float, float, float, float], conf: float, timestamp: float):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        if self._use_kalman:
            self.kf.update(np.array([cx, cy]))
            # optionally update bbox from state
        else:
            # compute simple velocity estimate
            dt = max(1e-3, timestamp - self.last_seen)
            self.vx = (cx - self.prev_cx) / dt
            self.vy = (cy - self.prev_cy) / dt
            self.prev_cx = cx
            self.prev_cy = cy

        self.bbox = bbox
        self.last_seen = timestamp
        self.hits += 1
        self.last_conf = float(conf)

    def get_velocity(self) -> Tuple[float, float, float]:
        if self._use_kalman:
            vx = float(self.kf.x[2, 0])
            vy = float(self.kf.x[3, 0])
        else:
            vx = float(self.vx)
            vy = float(self.vy)
        speed = math.hypot(vx, vy)
        return vx, vy, speed

    def time_since_creation(self, now_ts: float) -> float:
        return now_ts - self.created_at

    def time_since_seen(self, now_ts: float) -> float:
        return now_ts - self.last_seen


class Tracker:
    def __init__(self, max_age: float = 18.0, iou_threshold: float = 0.3):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    @staticmethod
    def _iou(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        a2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def update(self, detections: List[dict], timestamp: float = None):
        """Update tracker with current detections.

        detections: list of {'bbox': (x1,y1,x2,y2), 'label': str, 'conf': float}
        """
        if timestamp is None:
            timestamp = time.time()

        # Predict existing tracks
        for tr in self.tracks:
            tr.predict()

        if len(self.tracks) == 0:
            # create tracks for all detections
            for det in detections:
                self.tracks.append(Track(self.next_id, det['bbox'], det.get('label', ''), det.get('conf', 1.0), timestamp, max_age=self.max_age))
                self.next_id += 1
            return

        # Build cost matrix using 1 - IoU (lower = better). Infinity cost for different labels.
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros((N, M), dtype=float) + 1.0
        for i, tr in enumerate(self.tracks):
            for j, det in enumerate(detections):
                if tr.label != det.get('label'):
                    cost[i, j] = 1.0  # high cost if labels don't match
                    continue
                cost[i, j] = 1.0 - self._iou(tr.bbox, det['bbox'])

        assigned_tracks = set()
        assigned_dets = set()
        if HAVE_SCIPY and linear_sum_assignment is not None:
            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                if r < N and c < M and cost[r, c] <= (1.0 - self.iou_threshold):
                    # match
                    self.tracks[r].update(detections[c]['bbox'], detections[c].get('conf', 1.0), timestamp)
                    assigned_tracks.add(r)
                    assigned_dets.add(c)
        else:
            # Fallback greedy assignment: for each detection find best track
            for j, det in enumerate(detections):
                best_i = None
                best_cost = 1.0
                for i, tr in enumerate(self.tracks):
                    if i in assigned_tracks:
                        continue
                    if tr.label != det.get('label'):
                        continue
                    c = 1.0 - self._iou(tr.bbox, det['bbox'])
                    if c < best_cost:
                        best_cost = c
                        best_i = i
                if best_i is not None and best_cost <= (1.0 - self.iou_threshold):
                    self.tracks[best_i].update(det['bbox'], det.get('conf', 1.0), timestamp)
                    assigned_tracks.add(best_i)
                    assigned_dets.add(j)

        # unmatched detections -> create new tracks
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                self.tracks.append(Track(self.next_id, det['bbox'], det.get('label', ''), det.get('conf', 1.0), timestamp, max_age=self.max_age))
                self.next_id += 1

        # remove old tracks
        alive = []
        for tr in self.tracks:
            if timestamp - tr.last_seen <= tr.max_age:
                alive.append(tr)
        self.tracks = alive

    def get_active_tracks(self) -> List[Track]:
        return list(self.tracks)
