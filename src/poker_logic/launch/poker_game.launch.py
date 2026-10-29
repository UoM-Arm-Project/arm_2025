"""Decision + skill layer on top of poker_bringup/poker_arm.launch.py.

    ros2 launch poker_logic poker_game.launch.py mode:=dealer_player
    ros2 launch poker_logic poker_game.launch.py mode:=player dry_run:=true
    ros2 run poker_logic table_console          # in another terminal (stdin)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("poker_logic")
    args = [
        DeclareLaunchArgument("mode", default_value="player",
                              description="player | dealer | dealer_player"),
        DeclareLaunchArgument("seats", default_value="5"),
        DeclareLaunchArgument("bot_seat", default_value="0"),
        DeclareLaunchArgument("game_config", default_value=os.path.join(share, "config", "game.yaml")),
        DeclareLaunchArgument("layout_file", default_value=os.path.join(share, "config", "arm_layout.yaml")),
        DeclareLaunchArgument("dry_run", default_value="false",
                              description="log arm motions instead of sending /move_pose"),
    ]
    game_manager = Node(
        package="poker_logic", executable="game_manager", name="poker_game_manager", output="screen",
        parameters=[LaunchConfiguration("game_config"), {
            "mode": LaunchConfiguration("mode"),
            "seats": LaunchConfiguration("seats"),
            "bot_seat": LaunchConfiguration("bot_seat"),
        }])
    skill_executor = Node(
        package="poker_logic", executable="skill_executor", name="poker_skill_executor", output="screen",
        parameters=[{
            "layout_file": LaunchConfiguration("layout_file"),
            "seats": LaunchConfiguration("seats"),
            "dry_run": LaunchConfiguration("dry_run"),
        }])
    return LaunchDescription(args + [game_manager, skill_executor])
