"""3D Cubli controller: edge balance, jump-up face -> edge -> corner, corner balance.

Subscribes:  /cubli/state_estimate (std_msgs/Float64MultiArray)  see estimator3d_node.py
Publishes:   /cubli/wheel_torques  (std_msgs/Float64MultiArray)  3 motor torques [N m]
             /cubli/brake          (std_msgs/Int32MultiArray)    wheels to brake (jumps)
             /cubli/tilt           (std_msgs/Float64)  estimated tilt from corner balance [rad]
             /cubli/mode           (std_msgs/String, latched)
               FACE | SPIN_EDGE | JUMP_EDGE | EDGE | SPIN_CORNER | JUMP_CORNER | CORNER

The logic lives in cubli_control.control3d.JumpBalanceController (see its docstring);
this node only feeds it the estimate and passes on its torques and brake commands.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float64, Float64MultiArray, Int32MultiArray, String

from cubli_control.control3d import ControlParams, JumpBalanceController
from cubli_control.model3d import Cubli3DParams, tilt_angle
from cubli_control.ros_utils import spin_until_shutdown

TUPLES = ('edge_q', 'corner_q')


class Controller3D(Node):

    def __init__(self):
        super().__init__('cubli3d_controller')
        defaults = ControlParams()
        for name, value in vars(defaults).items():
            self.declare_parameter(name, list(value) if name in TUPLES else value)
        values = {}
        for name, value in vars(defaults).items():
            v = self.get_parameter(name).value
            values[name] = tuple(v) if name in TUPLES else type(value)(v)
        self.c = ControlParams(**values)
        self.p = Cubli3DParams(max_torque=self.c.max_torque)
        self.ctrl = JumpBalanceController(self.p, self.c)
        self.get_logger().info(
            'Corner LQR K =\n' + np.array2string(self.ctrl.k_corner, precision=4))
        self.get_logger().info('Edge LQR K = ' + np.array2string(self.ctrl.k_edge[0], precision=4))

        self.torque_pub = self.create_publisher(Float64MultiArray, 'cubli/wheel_torques', 10)
        self.brake_pub = self.create_publisher(Int32MultiArray, 'cubli/brake', 10)
        self.tilt_pub = self.create_publisher(Float64, 'cubli/tilt', 10)
        self.mode_pub = self.create_publisher(
            String, 'cubli/mode', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.logged = 0
        self.create_subscription(Float64MultiArray, 'cubli/state_estimate', self.on_state, 10)

    def on_state(self, msg: Float64MultiArray):
        if len(msg.data) != 9:
            return
        data = np.array(msg.data)
        up, rate, wheels = data[0:3], data[3:6], data[6:9]
        t = self.get_clock().now().nanoseconds * 1e-9
        torque, brake = self.ctrl.step(t, up, rate, wheels)
        if brake:
            self.brake_pub.publish(Int32MultiArray(data=[int(i) for i in brake]))
        self.torque_pub.publish(Float64MultiArray(data=[float(v) for v in torque]))
        self.tilt_pub.publish(Float64(data=tilt_angle(self.p, up)))
        for _, mode, note in self.ctrl.log[self.logged:]:
            self.mode_pub.publish(String(data=mode))
            self.get_logger().info(f'{mode}{"  " + note if note else ""}')
        self.logged = len(self.ctrl.log)


def main(args=None):
    rclpy.init(args=args)
    node = Controller3D()
    try:
        spin_until_shutdown(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
