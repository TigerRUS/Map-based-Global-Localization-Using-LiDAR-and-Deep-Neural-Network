#ifndef VISUALIZATION_UTILS_HPP
#define VISUALIZATION_UTILS_HPP

#include <vector>
#include <string>
#include <fstream>
#include <nav_msgs/msg/occupancy_grid.hpp>

namespace global_localization_cnn
{

class VisualizationUtils
{
public:
    static bool saveMapAsImage(const nav_msgs::msg::OccupancyGrid & map, 
                               const std::string & filename);
    
    static bool saveHeatmap(const std::vector<float> & data, 
                           int width, int height,
                           const std::string & filename);
    
    static std::vector<float> createGaussianKernel(int size, float sigma);
    
    static std::vector<float> applyGaussianBlur(const std::vector<float> & data,
                                                int width, int height,
                                                float sigma);
    
    static void printMapStatistics(const nav_msgs::msg::OccupancyGrid & map);
    
    static std::vector<std::vector<float>> generateDebugGrid(
        int width, int height, float resolution);
};

}

#endif