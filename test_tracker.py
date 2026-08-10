import os
import unittest
import numpy as np
import cv2
from tracker import CricketBallTracker, KalmanBallTracker, process_cricket_video, generate_sample_video


class TestTracker(unittest.TestCase):
    def setUp(self):
        self.sample_video = "data/raw/test_sample.mp4"
        self.output_video = "outputs/test_output.mp4"
        generate_sample_video(self.sample_video, num_frames=30)

    def tearDown(self):
        if os.path.exists(self.sample_video):
            os.remove(self.sample_video)
        if os.path.exists(self.output_video):
            os.remove(self.output_video)

    def test_detection(self):
        detector = CricketBallTracker()
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        # Draw red circle
        cv2.circle(frame, (150, 150), 10, (20, 20, 200), -1)
        res = detector.detect_ball(frame)
        self.assertIsNotNone(res)
        cx, cy, r, conf = res
        self.assertTrue(140 <= cx <= 160)
        self.assertTrue(140 <= cy <= 160)

    def test_kalman(self):
        kalman = KalmanBallTracker()
        kalman.init(100, 100)
        corrected = kalman.correct(105, 105)
        self.assertEqual(len(corrected), 2)

    def test_video_processing(self):
        metrics = process_cricket_video(self.sample_video, self.output_video)
        self.assertTrue(os.path.exists(self.output_video))
        self.assertGreater(metrics["total_frames"], 0)
        self.assertGreater(metrics["trajectory_points"], 0)
        self.assertIn("drs_pitching", metrics)
        self.assertIn("drs_impact", metrics)
        self.assertIn("drs_wickets", metrics)
        self.assertIn("drs_decision", metrics)

    def test_drs_projection(self):
        from tracker import calculate_drs_projection
        trajectory = [(640, 200, 1, True), (640, 300, 2, True), (640, 400, 3, True), (640, 480, 4, True)]
        drs = calculate_drs_projection(trajectory, (640, 300), (640, 480), 1280, 720)
        self.assertIn(drs["pitching_status"], ["IN-LINE", "OUTSIDE OFF", "OUTSIDE LEG"])
        self.assertIn(drs["impact_status"], ["IN-LINE", "OUTSIDE OFF", "OUTSIDE LEG"])
        self.assertGreater(len(drs["projected_points"]), 0)


if __name__ == "__main__":
    unittest.main()
