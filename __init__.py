import argparse
import queue
import sys
import threading
import time
from typing import List, Union

from calibration.CalibrationCommandSource import (
    CalibrationCommandSource,
    NTCalibrationCommandSource,
)
from calibration.CalibrationSession import CalibrationSession
from config.Config import ConfigStore, LocalConfig, RemoteConfig
from config.ConfigSource import ConfigSource, FileConfigSource, NTConfigSource
from config.VisionTypes import FiducialFrameObservation
from pipeline.Capture import GStreamerCapture
from pipeline.FiducialDetector import ArucoFiducialDetector
from pipeline.CameraPoseEstimator import MultiTargetCameraPoseEstimator
from output.OutputPublisher import NTOutputPublisher, OutputPublisher
from output.StreamServer import MjpegServer
from output.StreamServer import StreamServer
from output.OverlayUtil import overlay_frame_observation
from pipeline.Capture import CAPTURE_IMPLS 

import ntcore
from AprilTagWorker import apriltag_worker
import cv2 as cv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/Users/sim/Projects/visionPython/Vision/config.json")
    parser.add_argument("--calibration", default="/Users/sim/Projects/visionPython/Vision/calibration.json")
    args = parser.parse_args()

    config = ConfigStore(LocalConfig(), RemoteConfig())
    local_config_source: ConfigSource = FileConfigSource(args.config, args.calibration)
    remote_config_source: ConfigSource = NTConfigSource()
    calibration_command_source: CalibrationCommandSource = NTCalibrationCommandSource()
    local_config_source.update(config)

    capture = CAPTURE_IMPLS[config.local_config.capture_impl]()
    output_publisher: OutputPublisher = NTOutputPublisher()
    calibration_session = CalibrationSession()
    calibration_session_server: Union[StreamServer, None] = None

    if config.local_config.apriltags_enable:
        apriltag_worker_in = queue.Queue(maxsize=1)
        apriltag_worker_out = queue.Queue(maxsize=1)
        apriltag_worker = threading.Thread(
            target=apriltag_worker,
            args=(apriltag_worker_in, apriltag_worker_out, config.local_config.apriltags_stream_port),
            daemon=True
        )
        apriltag_worker.start()

    ntcore.NetworkTableInstance.getDefault().setServer(config.local_config.server_ip)
    ntcore.NetworkTableInstance.getDefault().startClient4(config.local_config.device_id)

    apriltags_frame_count = 0
    apriltags_last_print = 0
    was_calibrating = False
    last_frame_observations: List[FiducialFrameObservation] = []
    video_frame_cache: List[cv.Mat] = []
    while True:
        remote_config_source.update(config)
        timestamp = time.time()
        success, frame = capture.get_frame(config)

        # Exit if no frame
        if not success:
            time.sleep(0.5)
            continue

        if calibration_command_source.get_calibrating(config):
            # Calibration mode
            if not was_calibrating:
                calibration_session_server = MjpegServer()
                calibration_session_server.start(7999)
            was_calibrating = True
            calibration_session.process_frame(frame, calibration_command_source.get_capture_flag(config))
            calibration_session_server.set_frame(frame)

        elif was_calibrating:
            # Finish calibration
            calibration_session.finish()
            sys.exit(0)

        elif config.local_config.has_calibration:
            # AprilTag pipeline
            if config.local_config.apriltags_enable:
                try:
                    apriltag_worker_in.put((timestamp, frame, config), block=False)
                except: # No space in queue
                    pass
                try:
                    (
                        timestamp_out,
                        frame_observations,
                        camera_pose_observation,
                        tag_angle_observations,
                        demo_pose_observation,
                    ) = apriltag_worker_out.get(block=False)
                except: # No new frames
                    pass
                else:
                    # Publish observation
                    output_publisher.send_apriltag_observation(
                        config, timestamp_out, camera_pose_observation, tag_angle_observations, demo_pose_observation
                    )

                    # Store last observations
                    last_frame_observations = frame_observations

                    # Measure FPS
                    fps = None
                    apriltags_frame_count += 1
                    if time.time() - apriltags_last_print > 1:
                        apriltags_last_print = time.time()
                        print("Running AprilTag pipeline at", apriltags_frame_count, "fps")
                        output_publisher.send_apriltag_fps(config, timestamp_out, apriltags_frame_count)
                        apriltags_frame_count = 0

        else:
            # No calibration
            print("No calibration found")
            time.sleep(0.5)