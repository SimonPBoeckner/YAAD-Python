import json
from abc import ABC, abstractmethod

import cv2 as cv
import ntcore
import numpy as np

from config.Config import ConfigStore, RemoteConfig

class ConfigSource(ABC):
    """
    Abstract base class for configuration sources.
    This class defines the interface for updating a ConfigStore with configuration data.
    """

    @abstractmethod
    def update(self, config_store: ConfigStore) -> None:
        pass


class FileConfigSource(ConfigSource):
    """
    Configuration source that reads from a file.
    It reads configuration data from a JSON file and calibration data from a JSON file.
    """

    def __init__(self, config_filename: str, calibration_filename: str) -> None:
        self.config_filename = config_filename
        self.calibration_filename = calibration_filename
        pass

    def update(self, config_store: ConfigStore) ->  None:
        # Get config
        with open(self.config_filename, "r") as config_file:
            config_data = json.loads(config_file.read())
            config_store.local_config.device_id = config_data["device_id"]
            config_store.local_config.server_ip = config_data["server_ip"]
            config_store.local_config.apriltags_stream_port = config_data["apriltags_stream_port"]
            config_store.local_config.capture_impl = config_data["capture_impl"]
            config_store.local_config.apriltags_enable = config_data["apriltags_enable"]

        # Get calibration
        calibration_store = cv.FileStorage(self.calibration_filename, cv.FILE_STORAGE_READ)
        camera_matrix = calibration_store.getNode("camera_matrix").mat()
        distortion_coefficients = calibration_store.getNode("distortion_coefficients").mat()
        calibration_store.release()
        if type(camera_matrix) == np.ndarray and type(distortion_coefficients) == np.ndarray:
            config_store.local_config.camera_matrix = camera_matrix
            config_store.local_config.distortion_coefficients = distortion_coefficients
            config_store.local_config.has_calibration = True


class NTConfigSource(ConfigSource):
    """
    Configuration source that reads from NetworkTables.
    It subscribes to various topics to get remote configuration data.
    """

    init_complete: bool = False
    camera_id_sub: ntcore.StringSubscriber
    camera_resolution_width_sub: ntcore.IntegerSubscriber
    camera_resolution_height_sub: ntcore.IntegerSubscriber
    camera_framerate_sub: ntcore.DoubleSubscriber
    camera_auto_exposure_sub: ntcore.IntegerSubscriber
    camera_exposure_sub: ntcore.IntegerSubscriber
    camera_gain_sub: ntcore.DoubleSubscriber
    camera_denoise_sub: ntcore.DoubleSubscriber
    fiducial_size_m_sub: ntcore.DoubleSubscriber
    tag_layout_sub: ntcore.StringSubscriber
    timestamp_sub: ntcore.IntegerSubscriber

    def update(self, config_store: ConfigStore) -> None:
        # Initialize subscribers on first call
        if not self.init_complete:
            nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
                "/" + config_store.local_config.device_id + "/config"
            )
            self.camera_id_sub = nt_table.getStringTopic("camera_id").subscribe(RemoteConfig.camera_id)
            self.camera_resolution_width_sub = nt_table.getIntegerTopic("camera_resolution_width").subscribe(
                RemoteConfig.camera_resolution_width
            )
            self.camera_resolution_height_sub = nt_table.getIntegerTopic("camera_resolution_height").subscribe(
                RemoteConfig.camera_resolution_height
            )
            self.camera_framerate_sub = nt_table.getDoubleTopic("camera_framerate").subscribe(
                RemoteConfig.camera_framerate
            )
            self.camera_auto_exposure_sub = nt_table.getIntegerTopic("camera_auto_exposure").subscribe(
                RemoteConfig.camera_auto_exposure
            )
            self.camera_exposure_sub = nt_table.getIntegerTopic("camera_exposure").subscribe(
                RemoteConfig.camera_exposure
            )
            self.camera_gain_sub = nt_table.getDoubleTopic("camera_gain").subscribe(RemoteConfig.camera_gain)
            self.camera_denoise_sub = nt_table.getDoubleTopic("camera_denoise").subscribe(RemoteConfig.camera_denoise)
            self.fiducial_size_m_sub = nt_table.getDoubleTopic("fiducial_size_m").subscribe(
                RemoteConfig.fiducial_size_m
            )
            self.tag_layout_sub = nt_table.getStringTopic("tag_layout").subscribe("")
            self.timestamp_sub = nt_table.getIntegerTopic("timestamp").subscribe(0)
            self.init_complete = True

        # Read config data
        config_store.remote_config.camera_id = self.camera_id_sub.get()
        config_store.remote_config.camera_resolution_width = self.camera_resolution_width_sub.get()
        config_store.remote_config.camera_resolution_height = self.camera_resolution_height_sub.get()
        config_store.remote_config.camera_framerate = self.camera_framerate_sub.get()
        config_store.remote_config.camera_auto_exposure = self.camera_auto_exposure_sub.get()
        config_store.remote_config.camera_exposure = self.camera_exposure_sub.get()
        config_store.remote_config.camera_gain = self.camera_gain_sub.get()
        config_store.remote_config.camera_denoise = self.camera_denoise_sub.get()
        config_store.remote_config.fiducial_size_m = self.fiducial_size_m_sub.get()
        try:
            config_store.remote_config.tag_layout = json.loads(self.tag_layout_sub.get())
        except:
            with open("/Users/sim/Projects/visionPython/Vision/layout/2025-official.json", "r") as layout_file:
                config_store.remote_config.tag_layout = json.loads(layout_file.read())
            pass
        config_store.remote_config.timestamp = self.timestamp_sub.get()