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

    # FAST-LIO publishes 3dmap -> body with publish.heading_yaw already
    # applied, so body +X is the robot heading and the mesh follows it.
    # The sensor is 0.5157 m above the pelvis (waist 0.1157 + livox 0.4).
    # Assumes waist_yaw = 0.
    return [
        Node(
            package="fastlio_localization",
            executable="fastlio_mapping",
            name="laserMapping",
            output="screen",
            parameters=node_params,
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="body_to_pelvis",
            arguments=[
                "--x", "0",
                "--y", "0",
                "--z", "-0.5157",
                "--roll", "0",
                "--pitch", "0",
                "--yaw", "0",
                "--frame-id", "body",
                "--child-frame-id", "pelvis_link",
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config_path],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ]
def generate_launch_description():
    pkg_share = get_package_share_directory("fastlio_localization")
    default_config = os.path.join(pkg_share, "config", "mid360_relocalization.yaml")
    default_rviz = os.path.join(pkg_share, "rviz", "livox_relocal.rviz")
    # Filename under share/fastlio_localization/map/ (the node prepends that dir).
    default_map = "11_three_storey_house_F1.ply"



    return LaunchDescription([
        DeclareLaunchArgument(
            "config",
            default_value=default_config,
            description="Parameter YAML (includes preprocess.lidar_type)",
        ),
        DeclareLaunchArgument(
            "lidar_type",
            default_value="5",
            description=(
                "Overrides preprocess.lidar_type (1=LIVOX … 5=SIM xyz PointCloud2). "
                "5 matches vd_mujoco --use_lidar."
            ),
        ),
        DeclareLaunchArgument(
            "map_file_path",
            default_value=default_map,
            description=(
                "Prior map filename under share/fastlio_localization/map/. "
                "Default is the F1 house cloud."
            ),
        ),
        DeclareLaunchArgument("rviz", default_value="true", description="Start RViz2"),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=default_rviz,
            description="RViz2 config file",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
