#ifndef GLOBAL_LOCALIZATION_NODE_HPP
#define GLOBAL_LOCALIZATION_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <memory>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

#include "data_processor.hpp"

namespace global_localization_cnn
{

class GlobalLocalizationNode : public rclcpp::Node
{
public:
    explicit GlobalLocalizationNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~GlobalLocalizationNode() = default;

private:
    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    bool performInference();
    void publishPose(const std::vector<float> & output);
    bool loadModel(const std::string & model_path);

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    DataProcessor processor_;

    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::Session> ort_session_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;

    nav_msgs::msg::OccupancyGrid::SharedPtr latest_map_;
    sensor_msgs::msg::LaserScan::SharedPtr latest_scan_;
    
    rclcpp::TimerBase::SharedPtr inference_timer_;
    
    std::string model_path_;
    double confidence_threshold_;
    bool map_received_;
    bool model_loaded_;
};

} // namespace global_localization_cnn

#endif // GLOBAL_LOCALIZATION_NODE_HPP