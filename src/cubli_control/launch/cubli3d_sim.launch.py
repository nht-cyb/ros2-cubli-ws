"""Start the 3D Cubli simulator, estimator, controller, robot_state_publisher and RViz.

    ros2 launch cubli_control cubli3d_sim.launch.py                 # face -> edge -> corner
    ros2 launch cubli_control cubli3d_sim.launch.py start:=edge     # balance on an edge
    ros2 launch cubli_control cubli3d_sim.launch.py start:=corner   # balance on the corner
    ros2 launch cubli_control cubli3d_sim.launch.py jump_to_corner:=false   # stop on the edge
    ros2 launch cubli_control cubli3d_sim.launch.py rviz:=false controller:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    description_dir = get_package_share_directory('cubli_description')
    control_dir = get_package_share_directory('cubli_control')

    with open(os.path.join(description_dir, 'urdf', 'cubli_3d.urdf')) as f:
        robot_description = f.read()
    params_file = os.path.join(control_dir, 'config', 'cubli3d.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('controller', default_value='true'),
        DeclareLaunchArgument('start', default_value='face',
                              description='face (jump-up) | edge | corner'),
        DeclareLaunchArgument('jump_to_corner', default_value='true'),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='cubli_control',
            executable='sim3d_node',
            name='cubli3d_sim',
            parameters=[params_file, {'start': LaunchConfiguration('start')}],
            output='screen',
        ),
        Node(
            package='cubli_control',
            executable='estimator3d_node',
            name='cubli3d_estimator',
            parameters=[params_file],
            output='screen',
        ),
        Node(
            package='cubli_control',
            executable='controller3d_node',
            name='cubli3d_controller',
            parameters=[params_file, {'jump_to_corner': ParameterValue(
                LaunchConfiguration('jump_to_corner'), value_type=bool)}],
            output='screen',
            condition=IfCondition(LaunchConfiguration('controller')),
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(description_dir, 'rviz', 'cubli3d.rviz')],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
