"""Real OpenCV image checks plus clock semantics; no ROS/Gazebo claim."""

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from vision_core import ArucoReader, ObservationTracker

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None


TASK = {"marker_dictionary": "DICT_4X4_50",
        "marker_id_to_fruit": {7: "banana", 23: "apple", 42: "strawberry"}}


@unittest.skipUnless(cv2 is not None and hasattr(cv2, "aruco"),
                     "OpenCV ArUco is required for real image tests")
class MarkerImageTests(unittest.TestCase):
    def setUp(self):
        self.reader = ArucoReader(TASK)

    def image(self, marker_ids, side=100):
        image = np.full((480, 640, 3), 255, dtype=np.uint8)
        for index, marker_id in enumerate(marker_ids):
            if hasattr(cv2.aruco, "drawMarker"):
                marker = cv2.aruco.drawMarker(self.reader.dictionary, marker_id, side)
            else:
                marker = cv2.aruco.generateImageMarker(self.reader.dictionary, marker_id, side)
            x, y = 35 + index * 180, 100
            image[y:y + side, x:x + side] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        return image

    def test_supplied_practice_ids_decode_to_configured_labels(self):
        # Order in the picture differs from sorted JSON order.
        ids, labels, corners = self.reader.detect(self.image([42, 7, 23]))
        self.assertEqual(ids, [7, 23, 42])
        self.assertEqual(labels, ["banana", "apple", "strawberry"])
        self.assertEqual(len(corners), 3)

    def test_unknown_marker_is_reported_without_inventing_a_fruit(self):
        ids, labels, _ = self.reader.detect(self.image([13]))
        self.assertEqual(ids, [13])
        self.assertEqual(labels, [None])
        self.assertEqual(json.loads(json.dumps(labels)), [None])

    def test_small_marker_decodes_in_an_ideal_image(self):
        # About 40 pixels across at the practice observation distance. This
        # checks dictionary decoding, not Gazebo lighting or optical quality.
        ids, labels, _ = self.reader.detect(self.image([23], side=40))
        self.assertEqual((ids, labels), ([23], ["apple"]))

    def test_blank_image_does_not_reuse_the_previous_detection(self):
        self.assertEqual(self.reader.detect(self.image([7]))[0], [7])
        self.assertEqual(self.reader.detect(self.image([])), ([], [], []))

    def test_annotation_preserves_the_input_image(self):
        image = self.image([7])
        before = image.copy()
        ids, labels, corners = self.reader.detect(image)
        annotated = self.reader.annotate(image, ids, labels, corners)
        self.assertTrue(np.array_equal(image, before))
        self.assertFalse(np.array_equal(image, annotated))


class ObservationFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.tracker = ObservationTracker()

    def record(self, **changes):
        values = dict(source_stamp_ns=10_000_000_000,
                      frame_id="tb_test/camera_optical_frame", width=640, height=480,
                      ids=[7], labels=["banana"], now_mono=100.0, now_ns=10_100_000_000)
        values.update(changes)
        return self.tracker.update(**values)

    def test_new_blank_image_clears_the_previous_id_immediately(self):
        self.record()
        self.record(source_stamp_ns=10_200_000_000, now_ns=10_300_000_000,
                    now_mono=100.2, ids=[], labels=[])
        observation = self.tracker.snapshot(100.2, 10_300_000_000)
        self.assertEqual(observation["status"], "NO_MARKER")
        self.assertEqual(observation["ids"], [])
        self.assertEqual(observation["source_stamp_ns"], 10_200_000_000)

    def test_paused_simulation_does_not_pause_the_receipt_timeout(self):
        self.record()
        observation = self.tracker.snapshot(102.01, 10_100_000_000)
        self.assertEqual(observation["status"], "STALE")
        self.assertEqual((observation["ids"], observation["labels"]), ([], []))
        self.assertEqual(observation["source_age_s"], 0.1)
        self.assertEqual(observation["source_stamp_ns"], 10_000_000_000)

    def test_duplicate_or_older_image_stamps_do_not_extend_freshness(self):
        self.record()
        self.assertFalse(self.record(now_mono=101.5, now_ns=10_200_000_000))
        self.assertFalse(self.record(source_stamp_ns=9_900_000_000,
                                     now_mono=101.9, now_ns=10_300_000_000))
        observation = self.tracker.snapshot(102.1, 10_300_000_000)
        self.assertEqual(observation["status"], "STALE")
        self.assertEqual(observation["received_count"], 3)
        self.assertEqual(observation["received_age_s"], 2.1)

    def test_old_or_future_source_images_are_not_usable(self):
        self.assertFalse(self.record(source_stamp_ns=1_000_000_000))
        self.assertEqual(self.tracker.snapshot(100, 10_100_000_000)["status"], "STALE")
        self.assertFalse(self.record(source_stamp_ns=20_000_000_000))
        observation = self.tracker.snapshot(100, 10_100_000_000)
        self.assertEqual(observation["ids"], [])
        self.assertIn("ahead", observation["reason"])

    def test_clock_reset_clears_previous_run_and_accepts_new_timestamps(self):
        self.record()
        reset = self.tracker.snapshot(100.2, 500_000_000)
        self.assertEqual(reset["status"], "STALE")
        self.assertIsNone(reset["source_stamp_ns"])
        self.assertTrue(self.record(source_stamp_ns=600_000_000, now_ns=700_000_000,
                                    now_mono=100.3, ids=[23], labels=["apple"]))
        observation = self.tracker.snapshot(100.3, 700_000_000)
        self.assertEqual((observation["status"], observation["ids"]), ("LIVE", [23]))

    def test_failed_image_conversion_clears_visible_ids(self):
        self.record()
        self.record(source_stamp_ns=10_200_000_000, now_ns=10_300_000_000,
                    now_mono=100.2, error="image conversion failed")
        observation = self.tracker.snapshot(100.2, 10_300_000_000)
        self.assertEqual(observation["status"], "STALE")
        self.assertEqual(observation["ids"], [])

    def test_zero_stamp_and_missing_frame_id_fail_validation(self):
        self.assertFalse(self.record(source_stamp_ns=0))
        self.assertFalse(self.record(frame_id=""))
        self.assertEqual(self.tracker.snapshot(100, 10_100_000_000)["status"], "STALE")

    def test_json_keeps_unknown_label_and_contains_no_room_or_pose(self):
        self.record(ids=[13], labels=[None])
        observation = self.tracker.snapshot(100.1, 10_200_000_000)
        encoded = json.loads(json.dumps(observation, allow_nan=False))
        self.assertEqual(encoded["labels"], [None])
        self.assertNotIn("station", encoded)
        self.assertNotIn("pose", encoded)
        observation["ids"].append(42)
        self.assertEqual(self.tracker.snapshot(100.1, 10_200_000_000)["ids"], [13])




class ViewerModeTests(unittest.TestCase):
    def test_make_rviz_uses_the_display_overlay_and_build_includes_plugin(self):
        import subprocess

        recipe = subprocess.check_output(
            ["make", "--dry-run", "--no-print-directory", "build", "rviz",
             "ROBOT_NAME=tb_alpha", "ROS_DOMAIN_ID=36"], cwd=ROOT, text=True)
        self.assertIn("--packages-up-to planner mission rviz_aruco_camera", recipe)
        self.assertIn("source ros2_ws/install/setup.bash", recipe)
        self.assertIn("./launch/rviz.launch.py", recipe)
        self.assertIn('robot_name:=$ROBOT_NAME', recipe)
        self.assertIn('use_sim_time:=$USE_SIM_TIME', recipe)

    def test_camera_display_and_remap_use_the_selected_robot(self):
        import ast
        import json
        from pathlib import Path
        from string import Template
        from types import SimpleNamespace
        from unittest.mock import patch
        import yaml

        path = ROOT / "launch/rviz.launch.py"
        function, = [node for node in ast.parse(path.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name == "_launch_nodes"]
        namespace = dict(
            __file__=str(path), Path=Path, Template=Template, json=json,
            re=__import__("re"),
            tempfile=SimpleNamespace(TemporaryDirectory=lambda **options:
                                      SimpleNamespace(name="/tmp/rviz-test", cleanup=lambda: None)),
            LaunchConfiguration=lambda name: SimpleNamespace(perform=lambda context: context[name]),
            RegisterEventHandler=lambda *a: None, OnShutdown=lambda **kw: None,
            Node=lambda **kw: kw, ParameterValue=lambda value, value_type: value,
            _stop_session=None,
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        for robot, sim, task in (("tb_alpha", "true", "static/sim_task.yaml"),
                                 ("tb_beta", "false", "real_mission.yaml")):
            with self.subTest(robot=robot), patch.object(Path, "write_text") as write:
                _, node = namespace["_launch_nodes"](
                    dict(robot_name=robot, use_sim_time=sim, fixed_frame="map"))
                config = yaml.safe_load(write.call_args.args[0])
                manager = config["Visualization Manager"]
                display, = [item for item in manager["Displays"]
                            if item["Class"] == "rviz_aruco_camera/ArucoCamera"]
                self.assertTrue(display["Enabled"])
                alias = f"/{robot}/camera/video/ffmpeg"
                self.assertEqual(display["Topic"]["Value"], alias)
                self.assertEqual(display["Topic"]["Reliability Policy"], "Best Effort")
                self.assertEqual(display["Observations Topic"], f"/{robot}/vision/observations")
                self.assertEqual(display["Task Config"], str(ROOT / "config" / task))
                self.assertEqual(node["remappings"], [(alias, f"/{robot}/camera/video_h264")])
                self.assertEqual(node["parameters"][0][f"{robot}.camera.video.ffmpeg.decoders.h264"], "h264")
                self.assertNotIn("nav2_rviz_plugins/GoalTool", [tool["Class"] for tool in manager["Tools"]])


class ObservationCheckTests(unittest.TestCase):
    def test_requires_fresh_advancing_observations_with_one_frame(self):
        import json
        from aruco import ObservationCheck

        check = ObservationCheck(False)
        epoch = 1_700_000_000_000_000_000

        def receive(index, frame="front", status="NO_MARKER"):
            stamp = epoch + index * 100_000_000
            evidence = dict(clock="ros_system", clock_now_ns=stamp + 1,
                            source_stamp_ns=stamp, frame_id=frame,
                            image_width=1280, image_height=720,
                            received_age_s=0.01, status=status)
            return check.receive(json.dumps(evidence), index / 10, stamp + 1)

        self.assertIsNone(receive(1))
        self.assertIsNone(receive(1))
        self.assertIsNone(receive(2))
        self.assertIsNone(receive(3, status="STALE"))
        self.assertIsNone(receive(4))
        self.assertIsNone(receive(5, frame="new"))
        self.assertIsNone(receive(6, frame="new"))
        self.assertIsNotNone(receive(7, frame="new"))

class PreflightTests(unittest.TestCase):
    def test_sim_guard_and_turtlebot_smoke_are_separate(self):
        import os
        import subprocess
        from tempfile import TemporaryDirectory

        make = ["make", "--dry-run", "--no-print-directory",
                "-f", str(ROOT / "Makefile"),
                "ROBOT_NAME=tb_alpha", "ROS_DOMAIN_ID=36"]
        env = os.environ.copy()
        env.pop("USE_SIM_TIME", None)
        with TemporaryDirectory() as directory:
            for use_sim_time in (None, "true", "false"):
                with self.subTest(use_sim_time=use_sim_time):
                    mode = [] if use_sim_time is None else [f"USE_SIM_TIME={use_sim_time}"]
                    recipe = subprocess.check_output(
                        make + mode + ["ROS_LOCALHOST_ONLY=0", "rviz", "check"],
                        cwd=directory, env=env, text=True)
                    self.assertIn('--use-sim-time "$USE_SIM_TIME"', recipe)
                    self.assertNotIn("For SIM, set USE_SIM_TIME", recipe)
                    self.assertNotIn("scripts/check_hardware.py", recipe)
                    self.assertIn("--check-observations", recipe)

            raw = subprocess.check_output(
                make + ["CAMERA_TOPIC=/tb_alpha/custom", "check"], cwd=directory, env=env, text=True)
            self.assertIn('--image-topic "$CAMERA_TOPIC" --check', raw)
            self.assertNotIn("--check-observations", raw)
            sim = subprocess.check_output(
                make + ["sim"], cwd=directory, env=env, text=True)
            self.assertIn("For SIM, set USE_SIM_TIME", sim)
            smoke = subprocess.check_output(
                make + ["check-hardware"], cwd=directory, env=env, text=True)
        self.assertIn("scripts/check_hardware.py", smoke)
        self.assertNotIn("scripts/aruco.py", smoke)


if __name__ == "__main__":
    unittest.main()
