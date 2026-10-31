"""On-demand vision server (cards + pot services).

    ros2 launch poker_vision vision.launch.py
    ros2 launch poker_vision vision.launch.py config:=/path/to/vision.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_cfg = os.path.join(get_package_share_directory("poker_vision"), "config", "vision.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_cfg),
        Node(package="poker_vision", executable="vision_server", name="poker_vision_server",
             output="screen", parameters=[{"config_file": LaunchConfiguration("config")}]),
    ])
