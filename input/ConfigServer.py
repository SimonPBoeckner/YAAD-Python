import json
import time

import ntcore

CONFIG_PATH = "/Users/sim/Projects/vision/Vision/input/remote_config.json"

def load_config(path=CONFIG_PATH):
    with open(path, 'r') as file:
        return json.load(file)
    
def main():
    inst = ntcore.NetworkTableInstance.getDefault()
    inst.startServer()

    config = load_config()

    device_id = config.get("device_id")
    table = inst.getTable("/" + device_id + "/config")

    camera_id_pub = table.getStringTopic("camera_id").publish()
    camera_resolution_width_pub = table.getIntegerTopic("camera_resolution_width").publish()
    camera_resolution_height_pub = table.getIntegerTopic("camera_resolution_height").publish()
    camera_framerate_pub = table.getDoubleTopic("camera_framerate").publish()
    camera_auto_exposure_pub = table.getIntegerTopic("camera_auto_exposure").publish()
    camera_exposure_pub = table.getIntegerTopic("camera_exposure").publish()
    camera_gain_pub = table.getDoubleTopic("camera_gain").publish()
    camera_denoise_pub = table.getDoubleTopic("camera_denoise").publish()
    fiducial_size_m_pub = table.getDoubleTopic("fiducial_size_m").publish()

    while True:
        config = load_config()

        camera_id_pub.set(config.get("camera_id"))
        camera_resolution_width_pub.set(config.get("camera_resolution_width"))
        camera_resolution_height_pub.set(config.get("camera_resolution_height"))
        camera_framerate_pub.set(config.get("camera_framerate"))
        camera_auto_exposure_pub.set(config.get("camera_auto_exposure"))
        camera_exposure_pub.set(config.get("camera_exposure"))
        camera_gain_pub.set(config.get("camera_gain"))
        camera_denoise_pub.set(config.get("camera_denoise"))
        fiducial_size_m_pub.set(config.get("fiducial_size_m"))

        time.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
        print("Remote config server running. Press Ctrl+C to stop.")
    except KeyboardInterrupt:
        print("Stopping remote config server...")