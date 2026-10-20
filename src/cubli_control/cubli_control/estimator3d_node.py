"""State estimator for the 3D Cubli: raw IMU + three wheel encoders -> controller state.

Subscribes:  /imu/data_raw          (sensor_msgs/Imu)         body-frame axes
             /cubli/wheel_encoders  (sensor_msgs/JointState)  3 wheel angles [rad]
Publishes:   /cubli/state_estimate  (std_msgs/Float64MultiArray)
               [up_x, up_y, up_z, omega_x, omega_y, omega_z, wheel_x, wheel_y, wheel_z, t]:
               body-frame up vector, body rate [rad/s], wheel speeds [rad/s], and the
               IMU sample's stamp [s] (the controller runs on this clock)

The cube must be held still for the first `calib_samples` IMU samples (gyro bias
calibration); nothing is published until then, so the controller (and the sim) wait.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

from cubli_control.estimator3d import UpEstimator, WheelSpeeds
from cubli_control.ros_utils import spin_until_shutdown, warn_if_shared
from cubli_control.sensors3d import Sensor3DParams


class StateEstimator3D(Node):

    def __init__(self):
        super().__init__('cubli3d_estimator')
        self.declare_parameter('time_constant', 0.5)
        self.declare_parameter('imu_position', list(Sensor3DParams.imu_position))
        self.declare_parameter('calib_samples', 200)
        self.declare_parameter('wheel_speed_window', 8)

        gp = lambda name: self.get_parameter(name).value  # noqa: E731
        self.est = UpEstimator(time_constant=gp('time_constant'),
                               imu_position=tuple(gp('imu_position')),
                               calib_samples=int(gp('calib_samples')))
        self.wheels = WheelSpeeds(window=int(gp('wheel_speed_window')))
        self.last_imu_t = None

        self.pub = self.create_publisher(Float64MultiArray, 'cubli/state_estimate', 10)
        self.create_subscription(Imu, 'imu/data_raw', self.on_imu, 50)
        self.create_subscription(JointState, 'cubli/wheel_encoders', self.on_encoders, 50)
        warn_if_shared(self, ['cubli/state_estimate'])
        self.get_logger().info('Calibrating gyro bias: keep the cube still ...')

    def on_encoders(self, msg: JointState):
        if len(msg.position) == 3:
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.wheels.update(t, msg.position)

    def on_imu(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = 0.0 if self.last_imu_t is None else min(max(t - self.last_imu_t, 0.0), 0.05)
        self.last_imu_t = t
        w, a = msg.angular_velocity, msg.linear_acceleration
        gyro, accel = np.array([w.x, w.y, w.z]), np.array([a.x, a.y, a.z])

        if not self.est.ready:
            if self.est.calibrate(gyro, accel):
                self.get_logger().info(
                    f'Calibrated: gyro bias {np.round(self.est.bias, 4).tolist()} rad/s, '
                    f'up {np.round(self.est.up, 3).tolist()}')
            return
        up, rate = self.est.update(gyro, accel, dt)
        out = Float64MultiArray()
        out.data = [float(v) for v in np.concatenate([up, rate, self.wheels.speeds, [t]])]
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = StateEstimator3D()
    try:
        spin_until_shutdown(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
