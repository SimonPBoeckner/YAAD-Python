from dataclasses import dataclass
import cv2 as cv

@dataclass
class LocalConfig:
    device_id: str = "cam"
    server_ip: str = ""
    apriltags_stream_port: int = 8000
    capture_impl: str = ""
    apriltags_enable: bool = True
    has_calibration: bool = False
    camera_matrix: cv.typing.MatLike = None  # type: ignore
    distortion_coefficients: cv.typing.MatLike = None # type: ignore


@dataclass
class RemoteConfig:
    camera_id: str = ""
    camera_resolution_width: int = 0
    camera_resolution_height: int = 0
    camera_framerate: float = 0
    camera_auto_exposure: int = 0
    camera_exposure: int = 0
    camera_gain: float = 0
    camera_denoise: float = 0
    fiducial_size_m: float = 0
    tag_layout: any = None # type: ignore
    timestamp: int = 0


@dataclass
class ConfigStore:
    local_config: LocalConfig
    remote_config: RemoteConfig