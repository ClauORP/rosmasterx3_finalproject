import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Packages
    yahboom_nav_dir = get_package_share_directory('yahboomcar_nav')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    slam_gmapping_dir = get_package_share_directory('slam_gmapping')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Nav2 params
    params_file = LaunchConfiguration(
        'params_file',
        default=os.path.join(yahboom_nav_dir, 'params', 'rtabmap_nav_params.yaml')
    )

    map_yaml_path = LaunchConfiguration(
        'map',
        default=os.path.join(yahboom_nav_dir, 'maps', 'yahboomcar.yaml')
    )

    # --- Existing bringups ---
    laser_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(yahboom_nav_dir, 'launch', 'laser_bringup_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
        }.items(),
    )

    slam_gmapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_gmapping_dir, 'launch', 'slam_gmapping.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
        }.items(),
    )

    # --- Nav2 bringup (gives /navigate_to_pose) ---
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'map': map_yaml_path,
            'slam': 'False',
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value=use_sim_time,
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=params_file,
            description='Full path to Nav2 params file'
        ),
        DeclareLaunchArgument(
            'map',
            default_value=map_yaml_path,
            description='Full path to map yaml (still required by bringup_launch.py)'
        ),

        laser_bringup_launch,
        slam_gmapping_launch,
        nav2_launch,
    ])
