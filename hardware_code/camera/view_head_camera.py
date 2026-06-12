"""Standalone script to visualize head camera feed.

Press 'q' to quit.
"""

import cv2
from dexcontrol.config.vega import get_vega_config
from dexcontrol.robot import Robot
from loguru import logger

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360


def main():
    logger.info("Initializing robot with head camera...")

    configs = get_vega_config()
    configs.sensors.head_camera.enable = True
    configs.sensors.head_camera.use_rtc = False
    configs.heartbeat.enabled = False

    robot = Robot(configs=configs)

    if robot.sensors.head_camera.wait_for_active(timeout=10.0):
        logger.info("Head camera is active")
    else:
        logger.error("Head camera not active!")
        robot.shutdown()
        return

    logger.info("Press 'q' to quit")

    try:
        while True:
            head_obs = robot.sensors.head_camera.get_obs(obs_keys=["left_rgb"])
            head_image = head_obs.get("left_rgb")

            if head_image is not None:
                # Resize if needed
                if head_image.shape[0] != IMAGE_HEIGHT or head_image.shape[1] != IMAGE_WIDTH:
                    head_image = cv2.resize(head_image, (IMAGE_WIDTH, IMAGE_HEIGHT))

                # Convert RGB to BGR for OpenCV
                head_image_bgr = cv2.cvtColor(head_image, cv2.COLOR_RGB2BGR)
                cv2.imshow("Head Camera", head_image_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    finally:
        cv2.destroyAllWindows()
        robot.shutdown()
        logger.info("Done")


if __name__ == "__main__":
    main()
