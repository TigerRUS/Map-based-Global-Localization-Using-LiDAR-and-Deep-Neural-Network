from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    pkg_share = FindPackageShare('global_localization_cnn')
    
    config_path = PathJoinSubstitution([pkg_share, 'config', 'global_localization_params.yaml'])
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value=config_path,
            description='Path to config file'),
        
        Node(
            package='global_localization_cnn',
            executable='global_localization_node',
            name='global_localization_node',
            output='screen',
            parameters=[LaunchConfiguration('config')]
        )
    ])