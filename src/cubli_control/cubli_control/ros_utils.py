"""Small rclpy helpers shared by the nodes."""

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException


def spin_until_shutdown(node):
    """Spin until Ctrl+C / SIGTERM, without a traceback on the shutdown race.

    If the signal lands while spin() is building its next wait set, rclpy raises a plain
    RCLError (or a pybind11 RuntimeError) instead of ExternalShutdownException; that is a
    normal shutdown too.
    """
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except (RCLError, RuntimeError):
        if node.context.ok():
            raise


def warn_if_shared(node, topics):
    """Log an error while another node also publishes one of `topics` (e.g. a second
    Cubli launch in the same ROS domain): their data would mix and the cube misbehave."""
    def check():
        for topic in topics:
            count = node.count_publishers(topic)
            if count > 1:
                node.get_logger().error(
                    f'{count} publishers on {topic}: another Cubli launch is running in this '
                    'ROS domain, so the two mix their data. Stop the other one, or give each '
                    'its own ROS_DOMAIN_ID.', throttle_duration_sec=10.0)
    return node.create_timer(2.0, check)
