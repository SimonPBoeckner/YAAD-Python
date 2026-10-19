import cv2 as cv
import numpy as np
from config.Config import ConfigStore
from config.VisionTypes import FiducialFrameObservation, FiducialPoseObservation


def _tag_plane_object_points(fid_size_m: float) -> np.ndarray:
    h = fid_size_m / 2.0
    return np.array(
        [
            [-h, h, 0.0],
            [h, h, 0.0],
            [h, -h, 0.0],
            [-h, -h, 0.0],
        ],
        dtype=np.float64,
    )


def _draw_extruded_tag_box(
    frame: cv.Mat,
    observation: FiducialFrameObservation,
    config_store: ConfigStore,
    depth_m: float,
) -> None:
    cam = config_store.local_config.camera_matrix
    dist = config_store.local_config.distortion_coefficients
    if cam is None or dist is None:
        return
    fid = config_store.remote_config.fiducial_size_m
    if fid <= 0:
        return

    obj_plane = _tag_plane_object_points(fid)
    img_pts = observation.corners.reshape(4, 2).astype(np.float64)

    try:
        _, rvecs, tvecs, errors = cv.solvePnPGeneric(
            obj_plane,
            img_pts,
            cam,
            dist,
            flags=cv.SOLVEPNP_IPPE_SQUARE,
        )
    except cv.error:
        return

    if not rvecs or not tvecs:
        return

    best_i = 0
    if len(errors) > 1 and float(errors[1][0]) < float(errors[0][0]):
        best_i = 1
    rvec, tvec = rvecs[best_i], tvecs[best_i]

    h = fid / 2.0
    box_obj = np.array(
        [
            [-h, h, 0.0],
            [h, h, 0.0],
            [h, -h, 0.0],
            [-h, -h, 0.0],
            [-h, h, depth_m],
            [h, h, depth_m],
            [h, -h, depth_m],
            [-h, -h, depth_m],
        ],
        dtype=np.float64,
    )

    projected, _ = cv.projectPoints(box_obj, rvec, tvec, cam, dist)
    pts = projected.reshape(-1, 2).astype(int)

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
        line_color: int | tuple[int, int, int] = 255
    else:
        line_color = (0, 255, 100)
    for a, b in edges:
        cv.line(frame, pts[a], pts[b], line_color, 2, cv.LINE_AA)


def overlay_frame_observation(
    frame: cv.Mat,
    observation: FiducialFrameObservation,
    config_store: ConfigStore | None = None,
) -> None:
    cv.aruco.drawDetectedMarkers(frame, np.array([observation.corners]), np.array([observation.tag_id]))
    if config_store is not None:
        depth = config_store.remote_config.fiducial_size_m
        _draw_extruded_tag_box(frame, observation, config_store, depth)

def overlay_pose_observation(frame: cv.Mat, config_store: ConfigStore, observation: FiducialPoseObservation) -> None:
    cv.drawFrameAxes(
        frame,
        config_store.local_config.camera_matrix,
        config_store.local_config.distortion_coefficients,
        observation.rvec_0,
        observation.tvec_0,
        config_store.remote_config.fiducial_size_m / 2,
    )
    cv.drawFrameAxes(
        frame,
        config_store.local_config.camera_matrix,
        config_store.local_config.distortion_coefficients,
        observation.rvec_0,
        observation.tvec_1,
        config_store.remote_config.fiducial_size_m / 2,
    )