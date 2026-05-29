#include "global_localization_cnn/data_processor.hpp"
#include <cmath>
#include <algorithm>

namespace global_localization_cnn
{

std::vector<float> DataProcessor::normalizeMapData(const std::vector<int8_t> & data)
{
    std::vector<float> normalized(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] == -1) normalized[i] = 0.5f;
        else if (data[i] == 0) normalized[i] = 0.0f;
        else if (data[i] == 100) normalized[i] = 1.0f;
        else normalized[i] = data[i] / 100.0f;
    }
    return normalized;
}

std::vector<float> DataProcessor::processMap(const nav_msgs::msg::OccupancyGrid::SharedPtr & msg)
{
    map_info_ = msg->info;
    return normalizeMapData(msg->data);
}

std::vector<float> DataProcessor::processScan(
    const sensor_msgs::msg::LaserScan::SharedPtr & msg,
    const nav_msgs::msg::MapMetaData & map_info,
    tf2_ros::Buffer & tf_buffer)
{
    std::vector<float> scan_map(map_info.width * map_info.height, 0.0f);
    float angle = msg->angle_min;
    
    for (size_t i = 0; i < msg->ranges.size(); ++i) {
        float range = msg->ranges[i];
        
        if (range < msg->range_min || range > msg->range_max || 
            std::isnan(range) || std::isinf(range)) {
            angle += msg->angle_increment;
            continue;
        }
        
        float px = range * std::cos(angle);
        float py = range * std::sin(angle);
        
        try {
            geometry_msgs::msg::PointStamped point_laser;
            point_laser.header = msg->header;
            point_laser.point.x = px;
            point_laser.point.y = py;
            point_laser.point.z = 0.0;
            
            auto transform = tf_buffer.lookupTransform(
                "map", msg->header.frame_id, msg->header.stamp,
                rclcpp::Duration::from_seconds(1.0));
            
            geometry_msgs::msg::PointStamped point_map;
            tf2::doTransform(point_laser, point_map, transform);
            
            int u = static_cast<int>((point_map.point.x - map_info.origin.position.x) / map_info.resolution);
            int v = static_cast<int>((point_map.point.y - map_info.origin.position.y) / map_info.resolution);
            
            if (u >= 0 && u < static_cast<int>(map_info.width) && 
                v >= 0 && v < static_cast<int>(map_info.height)) {
                scan_map[v * map_info.width + u] = 1.0f;
            }
        } catch (const tf2::TransformException & e) {
            RCLCPP_WARN_STREAM(rclcpp::get_logger("data_processor"), "Transform error: " << e.what());
        }
        
        angle += msg->angle_increment;
    }
    
    return scan_map;
}

} // namespace global_localization_cnn