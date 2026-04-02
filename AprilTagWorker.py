from typing import Tuple, List, Union
from config.Config import ConfigStore
from output.OverlayUtil import overlay_frame_observation
from pipeline.FiducialDetector import ArucoFiducialDetector
from pipeline.CameraPoseEstimator import MultiTargetCameraPoseEstimator
from pipeline.TagAngleCalculator import CameraMatrixTagAngleCalculator
from config.VisionTypes import FiducialFrameObservation, FiducialPoseObservation, CameraPoseObservation, TagAngleObservation
from pipeline.PoseEstimator import SquareTargetPoseEstimator
from output.StreamServer import MjpegServer
import cv2 as cv
import queue

DEMO_ID = 42

def apriltag_worker(
        q_in: queue.Queue[Tuple[float, cv.Mat, ConfigStore]],
        q_out: queue.Queue[
            Tuple[
                float,
                List[FiducialFrameObservation],
                Union[CameraPoseObservation, None],
                List[TagAngleObservation],
                Union[FiducialPoseObservation, None],
            ]
        ],
        server_port: int,
):
    fiducial_detector = ArucoFiducialDetector(cv.aruco.DICT_APRILTAG_36H11)
    camera_pose_estimator = MultiTargetCameraPoseEstimator()
    tag_angle_calculator = CameraMatrixTagAngleCalculator()
    tag_pose_estimator = SquareTargetPoseEstimator()
    stream_server = MjpegServer()
    stream_server.start(server_port)

    while True:
        sample = q_in.get()
        timestamp: float = sample[0]
        frame: cv.Mat = sample[1]
        config: ConfigStore = sample[2]

        frame_observations = fiducial_detector.detect_fiducials(frame, config)
        camera_pose_observation = camera_pose_estimator.solve_camera_pose(
            [x for x in frame_observations if x.tag_id is not DEMO_ID],
            config
        )
        tag_angle_observations = [
            tag_angle_calculator.calculate_tag_angles(x, config) for x in frame_observations if x.tag_id is not DEMO_ID
        ]
        tag_angle_observations = [x for x in tag_angle_observations if x != None]
        demo_frame_observations = [x for x in frame_observations if x.tag_id == DEMO_ID]
        demo_pose_observation: Union[FiducialPoseObservation, None] = None
        if len(demo_frame_observations) > 0:
            demo_pose_observation = tag_pose_estimator.solve_fiducial_pose(
                demo_frame_observations[0], config
            )

        q_out.put(
            (timestamp, frame_observations, camera_pose_observation, tag_angle_observations, demo_pose_observation)
        )
        if stream_server.get_client_count() > 0:
            frame = frame.copy()
            
            [overlay_frame_observation(frame, x) for x in frame_observations]
            stream_server.set_frame(cv.cvtColor(frame, cv.COLOR_BGR2RGB))