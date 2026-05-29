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

## Configuration
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
