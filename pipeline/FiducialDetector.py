from abc import ABC, abstractmethod
from typing import List

import cv2 as cv

from config.Config import ConfigStore
from config.VisionTypes import FiducialFrameObservation

class FiducialDetector(ABC):
    """Abstract class for using various detectors to find fiducials in frames."""

    @abstractmethod
    def detect_fiducials(self, frame: cv.Mat, config_store: ConfigStore) -> List[FiducialFrameObservation]:
        pass


class ArucoFiducialDetector(FiducialDetector):
    """Detect Aruco fiducials in frames"""

    def __init__(self, dictionary_id) -> None:
        self.aruco_dict = cv.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_params = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def detect_fiducials(self, frame: cv.Mat, config_store: ConfigStore) -> List[FiducialFrameObservation]:
        corners, ids, _ = self.detector.detectMarkers(frame)
        if len(corners) == 0:
            return []
        return [FiducialFrameObservation(id[0], corner) for id, corner in zip(ids, corners)]