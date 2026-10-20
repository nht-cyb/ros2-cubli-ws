"""Physics simulator for the 3D Cubli: lying on a face, balancing on an edge or on the corner.

Subscribes:  /cubli/wheel_torques (std_msgs/Float64MultiArray)  3 motor torques [N m]
             /cubli/brake         (std_msgs/Int32MultiArray)    wheels to brake now (0, 1, 2)
             /cubli/push          (geometry_msgs/Vector3)  kick: adds this world-frame
                                                           angular velocity [rad/s]
Services:    /cubli/reset         (std_srvs/Trigger)  back to the start pose
Publishes:   /imu/data_raw         (sensor_msgs/Imu)         noisy ICM-42688-class IMU, body axes
             /cubli/wheel_encoders (sensor_msgs/JointState)  quantised wheel angles
             /joint_states         (sensor_msgs/JointState)  true wheel angles (for RViz)
             /tf                   world -> cube (true attitude, frame at the pivot corner)
             /cubli/ground_truth/tilt (std_msgs/Float64)     true tilt from corner balance [rad]

The plant is cubli_control.sim3d: contact physics (contact3d), Nidec motors with torque
error and ripple, lossy brakes, IMU and encoder models. Starting on an edge or the corner,
the cube is held still (as if by hand) until the controller's first command, which also
gives the estimator its still second; lying on a face it is still anyway.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Vector3
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64, Float64MultiArray, Int32MultiArray
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from cubli_control import contact3d
from cubli_control.model3d import (
    Cubli3DParams, initial_state, quat_to_matrix, tilt_angle, up_in_body)
from cubli_control.ros_utils import spin_until_shutdown
from cubli_control.sensors3d import Sensor3DParams
from cubli_control.sim3d import HardwareNoise, Simulator

WHEELS = ['wheel_x_joint', 'wheel_y_joint', 'wheel_z_joint']


class Cubli3DSim(Node):

    def __init__(self):
        super().__init__('cubli3d_sim')
        self.declare_parameter('rate_hz', 200.0)
        self.declare_parameter('substeps', 2)
        self.declare_parameter('start', 'face')        # face | edge | corner
        self.declare_parameter('start_tilt', 0.03)     # rad, for edge / corner starts
        self.declare_parameter('hold_time', 2.0)
        self.declare_parameter('command_timeout', 0.1)
        self.declare_parameter('start_timeout', 5.0)
        cube = Cubli3DParams()
        for name in ('max_torque', 'stall_torque', 'no_load_speed'):
            self.declare_parameter(name, getattr(cube, name))
        noise = HardwareNoise()
        for name in ('torque_scale', 'torque_noise_rel', 'torque_noise_abs',
                     'brake_efficiency', 'brake_efficiency_std'):
            self.declare_parameter(name, getattr(noise, name))
        sensors = Sensor3DParams()
        self.declare_parameter('imu_position', list(sensors.imu_position))
        self.declare_parameter('gyro_noise_std', sensors.gyro_noise_std)
        self.declare_parameter('gyro_bias', list(sensors.gyro_bias))
        self.declare_parameter('accel_noise_std', sensors.accel_noise_std)
        self.declare_parameter('accel_bias', list(sensors.accel_bias))
        self.declare_parameter('encoder_cpr', sensors.encoder_cpr)
        self.declare_parameter('seed', 0)

        gp = lambda name: self.get_parameter(name).value  # noqa: E731
        self.p = Cubli3DParams(max_torque=gp('max_torque'), stall_torque=gp('stall_torque'),
                               no_load_speed=gp('no_load_speed'))
        self.noise = HardwareNoise(
            torque_scale=gp('torque_scale'), torque_noise_rel=gp('torque_noise_rel'),
            torque_noise_abs=gp('torque_noise_abs'), brake_efficiency=gp('brake_efficiency'),
            brake_efficiency_std=gp('brake_efficiency_std'))
        self.sensor_params = Sensor3DParams(
            imu_position=tuple(gp('imu_position')), gyro_noise_std=gp('gyro_noise_std'),
            gyro_bias=tuple(gp('gyro_bias')), accel_noise_std=gp('accel_noise_std'),
            accel_bias=tuple(gp('accel_bias')), encoder_cpr=int(gp('encoder_cpr')))
        self.rate = gp('rate_hz')
        self.timeout = gp('command_timeout')
        self.hold_time = gp('hold_time')
        self.start_timeout = gp('start_timeout')

        self.command = np.zeros(3)
        self.pending_brake = []
        self.last_cmd = None
        self.hold_until = None
        self.start_time = self.now()
        self.reset_sim()

        self.imu_pub = self.create_publisher(Imu, 'imu/data_raw', 10)
        self.encoder_pub = self.create_publisher(JointState, 'cubli/wheel_encoders', 10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.tilt_pub = self.create_publisher(Float64, 'cubli/ground_truth/tilt', 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Float64MultiArray, 'cubli/wheel_torques', self.on_torques, 10)
        self.create_subscription(Int32MultiArray, 'cubli/brake', self.on_brake, 10)
        self.create_subscription(Vector3, 'cubli/push', self.on_push, 10)
        self.create_service(Trigger, 'cubli/reset', self.on_reset)
        self.create_timer(1.0 / self.rate, self.step)
        self.get_logger().info(
            f'3D Cubli sim at {self.rate:.0f} Hz: {self.p.mass:.2f} kg, start on a '
            f'{gp("start")}, max wheel torque {self.p.max_torque} N m, brake efficiency '
            f'{self.noise.brake_efficiency:.2f} +- {self.noise.brake_efficiency_std:.2f}')

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def start_state(self):
        start, tilt = self.get_parameter('start').value, self.get_parameter('start_tilt').value
        if start == 'corner':
            return initial_state(self.p, tilt, 0.5), True
        if start == 'edge':
            return contact3d.edge_state(self.p, 0, tilt), True
        return contact3d.face_state((0, 1)), False

    def reset_sim(self):
        x0, needs_hold = self.start_state()
        self.sim = Simulator(x0, p=self.p, sensors=self.sensor_params, noise=self.noise,
                             rate_hz=self.rate, substeps=int(self.get_parameter('substeps').value),
                             seed=int(self.get_parameter('seed').value))
        self.needs_hold = needs_hold

    def on_torques(self, msg: Float64MultiArray):
        if len(msg.data) != 3:
            return
        self.command = np.array(msg.data, float)
        self.last_cmd = self.now()
        if self.hold_until is None:
            self.hold_until = self.now()
            self.get_logger().info('Controller connected')

    def on_brake(self, msg: Int32MultiArray):
        wheels = [int(i) for i in msg.data if 0 <= int(i) < 3]
        if wheels:
            self.pending_brake = wheels

    def on_push(self, msg: Vector3):
        kick = np.array([msg.x, msg.y, msg.z])
        sim = self.sim
        sim.x[4:7] += quat_to_matrix(sim.x[0:4]).T @ kick
        if len(sim.contacts) == 1:   # on an edge only the rotation about it survives
            sim.x = contact3d._project(sim.x, self.p, sim.contacts)
        elif len(sim.contacts) >= 2:
            sim.x[4:7] = 0.0
        self.get_logger().info(f'Push: {np.round(kick, 3).tolist()} rad/s (world frame)')

    def on_reset(self, request, response):
        self.reset_sim()
        self.hold_until = self.now() + (self.hold_time if self.needs_hold else 0.0)
        response.success = True
        response.message = 'Reset to the start pose'
        return response

    def step(self):
        if not self.context.ok():
            return
        sim = self.sim
        if self.hold_until is None and self.now() - self.start_time > self.start_timeout:
            self.hold_until = self.now()
            self.get_logger().info('No controller command yet, letting go anyway')
        sim.held = self.needs_hold and (self.hold_until is None or self.now() < self.hold_until)
        command = self.command
        if self.last_cmd is None or self.now() - self.last_cmd > self.timeout:
            command = np.zeros(3)
        brake, self.pending_brake = self.pending_brake, []
        before = set(sim.contacts)
        sim.step(command, brake)
        if brake:
            eff = sim.brakes[-1][2]
            self.get_logger().info(f'Brake wheels {brake}: efficiency {eff:.3f}')
        if set(sim.contacts) != before:
            names = {0: 'corner', 1: 'edge', 2: 'face'}
            self.get_logger().info(f'Contact: {names[min(len(sim.contacts), 2)]} '
                                   f'{sorted(sim.contacts)}')
        self.publish()

    def publish(self):
        sim = self.sim
        stamp = self.get_clock().now().to_msg()
        q = sim.x[0:4]
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = 'world'
        tf.child_frame_id = 'cube'
        tf.transform.rotation.w, tf.transform.rotation.x = float(q[0]), float(q[1])
        tf.transform.rotation.y, tf.transform.rotation.z = float(q[2]), float(q[3])
        self.tf.sendTransform(tf)

        joints = JointState()
        joints.header.stamp = stamp
        joints.name = WHEELS
        joints.position = [float(v) for v in sim.x[10:13]]
        joints.velocity = [float(v) for v in sim.x[7:10]]
        joints.effort = [float(v) for v in sim.applied]
        self.joint_pub.publish(joints)
        self.tilt_pub.publish(Float64(data=tilt_angle(self.p, up_in_body(q))))

        gyro, accel, encoders = sim.read_sensors()
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = 'imu_link'
        imu.orientation_covariance[0] = -1.0   # no orientation estimate
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = map(float, gyro)
        (imu.linear_acceleration.x, imu.linear_acceleration.y,
         imu.linear_acceleration.z) = map(float, accel)
        self.imu_pub.publish(imu)

        enc = JointState()
        enc.header.stamp = stamp
        enc.name = WHEELS
        enc.position = [float(v) for v in encoders]
        self.encoder_pub.publish(enc)


def main(args=None):
    rclpy.init(args=args)
    node = Cubli3DSim()
    try:
        spin_until_shutdown(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
