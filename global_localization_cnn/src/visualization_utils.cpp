#include "global_localization_cnn/visualization_utils.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>

namespace global_localization_cnn
{

bool VisualizationUtils::saveMapAsImage(const nav_msgs::msg::OccupancyGrid & map, 
                                        const std::string & filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    
    file << "P2\n" << map.info.width << " " << map.info.height << "\n255\n";
    
    for (const auto & cell : map.data) {
        int pixel = 128;
        if (cell == -1) pixel = 128;
        else if (cell == 0) pixel = 255;
        else if (cell == 100) pixel = 0;
        else pixel = 255 - static_cast<int>(cell * 2.55);
        file << pixel << " ";
    }
    
    return true;
}

bool VisualizationUtils::saveHeatmap(const std::vector<float> & data, 
                                     int width, int height,
                                     const std::string & filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    
    file << "P2\n" << width << " " << height << "\n255\n";
    
    for (size_t i = 0; i < data.size(); ++i) {
        int pixel = static_cast<int>(std::clamp(data[i], 0.0f, 1.0f) * 255.0f);
        file << pixel << " ";
    }
    
    return true;
}

std::vector<float> VisualizationUtils::createGaussianKernel(int size, float sigma)
{
    std::vector<float> kernel(size * size);
    int center = size / 2;
    float sum = 0.0f;
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float x = i - center;
            float y = j - center;
            kernel[i * size + j] = std::exp(-(x*x + y*y) / (2.0f * sigma * sigma));
            sum += kernel[i * size + j];
        }
    }
    
    for (auto & val : kernel) val /= sum;
    return kernel;
}

std::vector<float> VisualizationUtils::applyGaussianBlur(const std::vector<float> & data,
                                                         int width, int height,
                                                         float sigma)
{
    int kernel_size = static_cast<int>(std::ceil(sigma * 3.0f)) * 2 + 1;
    auto kernel = createGaussianKernel(kernel_size, sigma);
    std::vector<float> result(data.size(), 0.0f);
    int half = kernel_size / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    int ny = std::clamp(y + ky, 0, height - 1);
                    int nx = std::clamp(x + kx, 0, width - 1);
                    sum += data[ny * width + nx] * kernel[(ky + half) * kernel_size + (kx + half)];
                }
            }
            result[y * width + x] = sum;
        }
    }
    
    return result;
}

void VisualizationUtils::printMapStatistics(const nav_msgs::msg::OccupancyGrid & map)
{
    int unknown = 0, free = 0, occupied = 0;
    
    for (const auto & cell : map.data) {
        if (cell == -1) unknown++;
        else if (cell == 0) free++;
        else if (cell == 100) occupied++;
    }
    
    std::cout << "Map Statistics:\n"
              << "  Resolution: " << map.info.resolution << " m/cell\n"
              << "  Size: " << map.info.width << "x" << map.info.height << "\n"
              << "  Origin: (" << map.info.origin.position.x << ", " 
              << map.info.origin.position.y << ")\n"
              << "  Free: " << free << " (" 
              << std::fixed << std::setprecision(1) 
              << (100.0 * free / map.data.size()) << "%)\n"
              << "  Occupied: " << occupied << " ("
              << (100.0 * occupied / map.data.size()) << "%)\n"
              << "  Unknown: " << unknown << " ("
              << (100.0 * unknown / map.data.size()) << "%)\n";
}

std::vector<std::vector<float>> VisualizationUtils::generateDebugGrid(
    int width, int height, float resolution)
{
    std::vector<std::vector<float>> grid(height, std::vector<float>(width, 0.0f));
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float px = x * resolution;
            float py = y * resolution;
            grid[y][x] = std::sin(px * 0.5f) * std::cos(py * 0.5f);
        }
    }
    
    return grid;
}

}