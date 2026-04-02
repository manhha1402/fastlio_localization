import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile


def _launch_setup(context, *args, **kwargs):
    config_path = LaunchConfiguration("config").perform(context)
    rviz_config_path = LaunchConfiguration("rviz_config").perform(context)

    node_params = [config_path]
    lidar_type_str = LaunchConfiguration("lidar_type").perform(context).strip()
    if lidar_type_str:
        node_params.append({"preprocess.lidar_type": int(lidar_type_str)})
    
    map_file_path_str = LaunchConfiguration("map_file_path").perform(context).strip()
    if map_file_path_str:
        node_params.append({"map_file_path": map_file_path_str})
    transform_lookup_broadcast_config = os.path.join(get_package_share_directory('vm_navigation_launch'), 'config', 'transform_lookup_and_broadcast.yaml')

    return [
        Node(
            package="fastlio_localization",
            executable="fastlio_mapping",
            name="laserMapping",
            output="screen",
            parameters=node_params,
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config_path],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
        # Node(
        # package='vm_navigation_launch',
        # executable='transform_lookup_and_broadcast',
        # name='transform_lookup_and_broadcast',
        # output='screen',
        # parameters=[ParameterFile(transform_lookup_broadcast_config, allow_substs=True)],
        #),
    ]
def generate_launch_description():
    pkg_share = get_package_share_directory("fastlio_localization")
    default_config = os.path.join(pkg_share, "config", "mid360_relocalization.yaml")
    default_rviz = os.path.join(pkg_share, "rviz", "livox_relocal.rviz")



    return LaunchDescription([
        DeclareLaunchArgument(
            "config",
            default_value=default_config,
            description="Parameter YAML (includes preprocess.lidar_type)",
        ),
        DeclareLaunchArgument(
            "lidar_type",
            default_value="",
            description=(
                "If set, overrides preprocess.lidar_type (1=LIVOX … 5=SIM). "
                "Empty = use YAML."
            ),
        ),
        DeclareLaunchArgument(
            "map_file_path",
            default_value="",
            description="If set, overrides map_file_path. Empty = use YAML.",
        ),
        DeclareLaunchArgument("rviz", default_value="true", description="Start RViz2"),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=default_rviz,
            description="RViz2 config file",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
