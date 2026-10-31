"""Overhead card detector + side pot estimator.

    ros2 launch poker_vision vision.launch.py
    ros2 launch poker_vision vision.launch.py pot:=false config:=/path/to/vision.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_cfg = os.path.join(get_package_share_directory("poker_vision"), "config", "vision.yaml")
    cfg = LaunchConfiguration("config")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_cfg),
        DeclareLaunchArgument("cards", default_value="true"),
        DeclareLaunchArgument("pot", default_value="true"),
        Node(package="poker_vision", executable="card_detector", name="poker_card_detector",
             output="screen", parameters=[{"config_file": cfg}],
             condition=IfCondition(LaunchConfiguration("cards"))),
        Node(package="poker_vision", executable="pot_estimator", name="poker_pot_estimator",
             output="screen", parameters=[{"config_file": cfg}],
             condition=IfCondition(LaunchConfiguration("pot"))),
    ])
