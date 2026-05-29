#ifndef DATA_PROCESSOR_HPP
#define DATA_PROCESSOR_HPP

#include <vector>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_ros/buffer.h>

namespace global_localization_cnn
{

class DataProcessor
{
public:
    DataProcessor() = default;
    ~DataProcessor() = default;

    std::vector<float> processMap(const nav_msgs::msg::OccupancyGrid::SharedPtr & msg);
    
    std::vector<float> processScan(
        const sensor_msgs::msg::LaserScan::SharedPtr & msg,
        const nav_msgs::msg::MapMetaData & map_info,
        tf2_ros::Buffer & tf_buffer);

    const nav_msgs::msg::MapMetaData & getMapInfo() const { return map_info_; }

private:
    std::vector<float> normalizeMapData(const std::vector<int8_t> & data);
    nav_msgs::msg::MapMetaData map_info_;
};

}

#endif