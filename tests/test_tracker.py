import unittest

from tracker import Tracker


class TrackerTests(unittest.TestCase):
    def test_matching_detection_keeps_track_id_and_increments_hits(self):
        tracker = Tracker()
        detection = {"bbox": (10.0, 10.0, 40.0, 40.0), "label": "cup", "conf": 0.9}

        tracker.update([detection], timestamp=1.0)
        tracker.update([detection], timestamp=2.0)

        tracks = tracker.get_active_tracks()
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].id, 1)
        self.assertEqual(tracks[0].hits, 2)

    def test_non_overlapping_detection_starts_a_new_track(self):
        tracker = Tracker()
        first = {"bbox": (0.0, 0.0, 20.0, 20.0), "label": "cup", "conf": 0.9}
        second = {"bbox": (100.0, 100.0, 120.0, 120.0), "label": "cup", "conf": 0.9}

        tracker.update([first], timestamp=1.0)
        tracker.update([second], timestamp=2.0)

        self.assertEqual([track.id for track in tracker.get_active_tracks()], [1, 2])


if __name__ == "__main__":
    unittest.main()
