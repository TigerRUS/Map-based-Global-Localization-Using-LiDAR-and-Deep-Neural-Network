#include "global_localization_cnn/global_localization_node.hpp"
#include <cmath>

namespace global_localization_cnn
{

GlobalLocalizationNode::GlobalLocalizationNode(const rclcpp::NodeOptions & options)
: Node("global_localization_node", options), map_received_(false), model_loaded_(false)
{
    model_path_ = this->declare_parameter("model_path", "localization_model.onnx");
    confidence_threshold_ = this->declare_parameter("confidence_threshold", 0.5);
    std::string map_topic = this->declare_parameter("map_topic", "/map");
    std::string scan_topic = this->declare_parameter("scan_topic", "/scan");
    std::string pose_topic = this->declare_parameter("initial_pose_topic", "/initialpose");
    double inference_rate = this->declare_parameter("inference_rate", 10.0);

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        map_topic, rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable(),
        std::bind(&GlobalLocalizationNode::mapCallback, this, std::placeholders::_1));

    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        scan_topic, rclcpp::SensorDataQoS(),
        std::bind(&GlobalLocalizationNode::scanCallback, this, std::placeholders::_1));

    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        pose_topic, 10);

    if (!loadModel(model_path_)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load ONNX model");
        return;
    }

    auto period = std::chrono::duration<double>(1.0 / inference_rate);
    inference_timer_ = this->create_wall_timer(
        period, std::bind(&GlobalLocalizationNode::performInference, this));

    RCLCPP_INFO(this->get_logger(), "Global Localization Node initialized");
}

void GlobalLocalizationNode::mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
    latest_map_ = msg;
    map_received_ = true;
    RCLCPP_INFO(this->get_logger(), "Map received: %dx%d", msg->info.width, msg->info.height);
}

void GlobalLocalizationNode::scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    latest_scan_ = msg;
}

bool GlobalLocalizationNode::loadModel(const std::string & model_path)
{
    try {
        ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "GlobalLocalization");
        Ort::SessionOptions session_opts;
        session_opts.SetIntraOpNumThreads(1);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        ort_session_ = std::make_unique<Ort::Session>(*ort_env_, model_path.c_str(), session_opts);
        
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t num_inputs = ort_session_->GetInputCount();
        input_names_.resize(num_inputs);
        for (size_t i = 0; i < num_inputs; ++i) {
            input_names_[i] = ort_session_->GetInputName(i, allocator);
        }
        
        size_t num_outputs = ort_session_->GetOutputCount();
        output_names_.resize(num_outputs);
        for (size_t i = 0; i < num_outputs; ++i) {
            output_names_[i] = ort_session_->GetOutputName(i, allocator);
        }
        
        auto input_info = ort_session_->GetInputTypeInfo(0);
        auto tensor_info = input_info.GetTensorTypeAndShapeInfo();
        input_shape_ = tensor_info.GetShape();
        
        if (input_shape_.size() == 4 && input_shape_[0] == -1) {
            input_shape_[0] = 1;
        }
        
        RCLCPP_INFO(this->get_logger(), "Model loaded: %s", model_path.c_str());
        model_loaded_ = true;
        return true;
    } catch (const Ort::Exception & e) {
        RCLCPP_ERROR(this->get_logger(), "ONNX Error: %s", e.what());
        return false;
    }
}

bool GlobalLocalizationNode::performInference()
{
    if (!model_loaded_ || !map_received_ || !latest_scan_) {
        return false;
    }
    
    if (!latest_map_ || !latest_scan_) {
        return false;
    }
    
    try {
        auto map_data = processor_.processMap(latest_map_);
        auto scan_data = processor_.processScan(latest_scan_, latest_map_->info, *tf_buffer_);
        
        int h = latest_map_->info.height;
        int w = latest_map_->info.width;
        
        std::vector<float> input_data(h * w * 2);
        std::copy(map_data.begin(), map_data.end(), input_data.begin());
        std::copy(scan_data.begin(), scan_data.end(), input_data.begin() + h * w);
        
        std::vector<int64_t> actual_shape = {1, 2, h, w};
        
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(), actual_shape.data(), actual_shape.size());
        
        std::vector<Ort::Value> output_tensors = ort_session_->Run(
            Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor, 1,
            output_names_.data(), output_names_.size());
        
        float * output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> output(output_data, output_data + 4);
        
        if (output[2] * output[2] + output[3] * output[3] > confidence_threshold_) {
            publishPose(output);
            return true;
        }
        
        return false;
    } catch (const Ort::Exception & e) {
        RCLCPP_ERROR(this->get_logger(), "Inference error: %s", e.what());
        return false;
    }
}

void GlobalLocalizationNode::publishPose(const std::vector<float> & output)
{
    auto pose_msg = geometry_msgs::msg::PoseWithCovarianceStamped();
    pose_msg.header.stamp = this->now();
    pose_msg.header.frame_id = "map";
    pose_msg.pose.pose.position.x = output[0];
    pose_msg.pose.pose.position.y = output[1];
    pose_msg.pose.pose.position.z = 0.0;
    pose_msg.pose.pose.orientation.x = 0.0;
    pose_msg.pose.pose.orientation.y = 0.0;
    pose_msg.pose.pose.orientation.z = output[2];
    pose_msg.pose.pose.orientation.w = output[3];
    
    pose_msg.pose.covariance[0] = 0.25;
    pose_msg.pose.covariance[7] = 0.25;
    pose_msg.pose.covariance[35] = 0.0685;
    
    pose_pub_->publish(pose_msg);
    RCLCPP_INFO(this->get_logger(), "Published pose: x=%.3f, y=%.3f, yaw=%.2f",
                output[0], output[1], std::atan2(output[2], output[3]) * 180.0 / M_PI);
}

} // namespace global_localization_cnn

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(global_localization_cnn::GlobalLocalizationNode)