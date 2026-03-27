import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("fastlio_localization")
    default_config = os.path.join(pkg_share, "config", "mid360_relocalization.yaml")
    default_rviz = os.path.join(pkg_share, "rviz", "livox_relocal.rviz")

    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config, description="Parameter YAML"),
        DeclareLaunchArgument("rviz", default_value="true", description="Start RViz2"),
        DeclareLaunchArgument("rviz_config", default_value=default_rviz, description="RViz2 config file"),
        Node(
            package="fastlio_localization",
            executable="fastlio_mapping",
            name="laserMapping",
            output="screen",
            parameters=[LaunchConfiguration("config")],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", LaunchConfiguration("rviz_config")],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
