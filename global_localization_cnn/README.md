Global Localization CNN

CNN-based global localization node for ROS2 Humble using ONNX Runtime. Robot global localization by matching LiDAR scans with occupancy grid maps.

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
Parameters file config/global_localization_params.yaml
- `model_path: "models/localization_model.onnx"` *Path to ONNX model*
- `confidence_threshold: 0.5` *Minimum confidence to publish pose*
- `map_topic: "/map"` *Occupancy grid topic*
- `scan_topic: "/scan"` *Laser scan topic*
- `initial_pose_topic: "/initialpose"` *Output pose topic*
- `inference_rate: 5.0` *Inference frequency (Hz)*
- `use_sim_time: false` *Use simulation time*