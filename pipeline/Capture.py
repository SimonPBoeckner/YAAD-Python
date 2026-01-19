from abc import ABC, abstractmethod
from typing import Tuple
import cv2 as cv
import time
from config.Config import ConfigStore
import dataclasses
import sys
import numpy

class Capture(ABC):
    """Abstract class for getting camera frames from various backends."""

    @abstractmethod
    def get_frame(self) -> Tuple[bool, cv.typing.MatLike]:
        """Return the next frame from the camera."""
        pass

    @staticmethod
    def config_changed(config_a: ConfigStore, config_b: ConfigStore) -> bool:
        if config_a == None and config_b ==  None:
            return False
        if config_a == None or config_b == None:
            return True
        
        remote_a = config_a.remote_config
        remote_b = config_b.remote_config

        return (
            remote_a.camera_id != remote_b.camera_id
            or remote_a.camera_resolution_width != remote_b.camera_resolution_width
            or remote_a.camera_resolution_height != remote_b.camera_resolution_height
            or remote_a.camera_auto_exposure != remote_b.camera_auto_exposure
            or remote_a.camera_exposure != remote_b.camera_exposure
            or remote_a.camera_gain != remote_b.camera_gain
            or remote_a.camera_denoise != remote_b.camera_denoise
        )


class DefaultCapture(Capture):
    """Read from camera using default OpenCV config"""

    def __init__(self) -> None:
        pass

    video = None
    last_config: ConfigStore

    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv.typing.MatLike]:
        if self.video != None and self.config_changed(self.last_config, config_store):
            print("Restarting capture session")
            self.video.release()
            self.video = None

        if self.video == None:
            self.video = cv.VideoCapture(1)
            self.video.set(cv.CAP_PROP_FRAME_WIDTH, config_store.remote_config.camera_resolution_width)
            self.video.set(cv.CAP_PROP_FRAME_HEIGHT, config_store.remote_config.camera_resolution_height)
            self.video.set(cv.CAP_PROP_AUTO_EXPOSURE, config_store.remote_config.camera_auto_exposure)
            self.video.set(cv.CAP_PROP_EXPOSURE, config_store.remote_config.camera_exposure)
            self.video.set(cv.CAP_PROP_GAIN, int(config_store.remote_config.camera_gain))

        self.last_config = config_store

        retval, frame = self.video.read()
        return retval, frame


class GStreamerCapture(Capture):
    """Read from camera using gstreamer config"""

    def __init__(self) -> None:
        pass

    video = None
    last_config: ConfigStore

    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv.typing.MatLike]:
        if self.video != None and self.config_changed(self.last_config, config_store):
            print("Config changed, stopping capture session")
            self.video.release()
            self.video = None
            time.sleep(2)

        if self.video == None:
            if config_store.remote_config.camera_id == "":
                print("No camera ID, waiting to start capture session")
            else:
                print("Starting capture session")
                self.video = cv.VideoCapture(
                    "v4l2src device=" 
                    + str(config_store.remote_config.camera_id) + ' extra-controls="c,exposure_auto=' 
                    + str(config_store.remote_config.camera_auto_exposure) + ",exposure_absolute=" 
                    + str(config_store.remote_config.camera_exposure) + ",gain=" 
                    + str(config_store.remote_config.camera_gain) + ',sharpness=0,brightness=0" ! image/jpeg format=MJPG, width=' 
                    + str(config_store.remote_config.camera_resolution_width) + ", height=" 
                    + str(config_store.remote_config.camera_resolution_height) + ", framerate=" 
                    + str(config_store.remote_config.camera_framerate) + "/1 ! jpegdec ! videoconvert ! video/x-raw, format=GRAY8 ! appsink drop=true sync=false max-buffers=1",
                    cv.CAP_GSTREAMER,
                )
                print("Capture session ready")
                
        self.last_config = ConfigStore(
            dataclasses.replace(config_store.local_config), dataclasses.replace(config_store.remote_config)
        )
        
        if self.video != None:
            retval, frame = self.video.read()
            if not retval:
                print("Capture session failed, restarting")
                self.video.release()
                self.video = None # Force reconnect
                sys.exit(1)
            return retval, frame
        else:
            return False, cv.Mat(numpy.ndarray([]))

CAPTURE_IMPLS = {
    "": DefaultCapture,
    "gstreamer":GStreamerCapture
}