# Map-based-Global-Localization-Using-LiDAR-and-Deep-Neural-Network

CNN-based global localization node for ROS2 Humble using ONNX Runtime. Localizes robot by matching LiDAR scans with occupancy grid maps.

## Description

This package provides a ROS2 node that performs global localization using a convolutional neural network. It subscribes to map and laser scan topics, runs inference through an ONNX model, and publishes the estimated pose on `/initialpose` topic.

## Dependencies

### ROS2 Packages
- rclcpp
- nav_msgs
- sensor_msgs
- geometry_msgs
- tf2, tf2_ros, tf2_geometry_msgs

### External
- **ONNX Runtime**

Configuration
Parameters file config/global_localization_params.yaml:

yaml
/**:
  ros__parameters:
    model_path: "models/localization_model.onnx"
    confidence_threshold: 0.5
    map_topic: "/map"
    scan_topic: "/scan"
    initial_pose_topic: "/initialpose"
    inference_rate: 5.0
    use_sim_time: false
Parameter Description
Parameter	Default	Description
model_path	models/localization_model.onnx	Path to ONNX model
confidence_threshold	0.5	Minimum confidence to publish pose
map_topic	/map	Occupancy grid topic
scan_topic	/scan	Laser scan topic
initial_pose_topic	/initialpose	Output pose topic
inference_rate	5.0	Inference frequency (Hz)
use_sim_time	false	Use simulation time
Required Systems
Before running the node, ensure these systems are running:

1. Map Server (Nav2)
bash
# Using Nav2 map server
ros2 run nav2_map_server map_server --ros-args -p map_file:=/path/to/map.yaml
Or launch complete Nav2:

bash
ros2 launch nav2_bringup navigation_launch.py map:=/path/to/map.yaml
2. Robot Bringup
bash
# Example for TurtleBot3
ros2 launch turtlebot3_bringup robot.launch.py

# Or your robot's specific bringup
ros2 launch your_robot bringup.launch.py
3. Transform Publisher
Ensure TF tree includes map → odom → base_link transforms (provided by robot_state_publisher and odometry source).

Usage
Launch Node
bash
# Using launch file
ros2 launch global_localization_cnn global_localization_launch.py

# Or run node directly
ros2 run global_localization_cnn global_localization_node
With Custom Parameters
bash
ros2 launch global_localization_cnn global_localization_launch.py config:=/path/to/your_params.yaml