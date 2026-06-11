#ifndef CONFIG_MANAGER_HPP
#define CONFIG_MANAGER_HPP

#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <optional>

namespace global_localization_cnn
{

class ConfigManager
{
public:
    struct ModelConfig {
        std::string model_path = "localization_model.onnx";
        float confidence_threshold = 0.5f;
        int batch_size = 1;
        bool use_gpu = false;
        int num_threads = 1;
        std::vector<int64_t> input_shape;
        std::string input_layer_name;
        std::string output_layer_name;
    };
    
    struct SensorConfig {
        std::string scan_topic = "/scan";
        std::string map_topic = "/map";
        std::string pose_topic = "/initialpose";
        float min_range = 0.1f;
        float max_range = 30.0f;
        int scan_queue_size = 10;
    };
    
    struct FilterConfig {
        bool enable_map_dilation = false;
        int dilation_size = 3;
        bool enable_scan_smoothing = false;
        float scan_smoothing_sigma = 1.0f;
        bool enable_outlier_rejection = false;
        float outlier_threshold = 0.1f;
    };
    
    ConfigManager() = default;
    
    bool loadFromYAML(const std::string & filename);
    bool saveToYAML(const std::string & filename) const;
    bool loadFromJSON(const std::string & filename);
    void setDefaults();
    
    ModelConfig& getModelConfig() { return model_config_; }
    SensorConfig& getSensorConfig() { return sensor_config_; }
    FilterConfig& getFilterConfig() { return filter_config_; }
    
    std::string toString() const;
    
private:
    ModelConfig model_config_;
    SensorConfig sensor_config_;
    FilterConfig filter_config_;
    
    std::string escapeYAML(const std::string & str) const;
    std::string trim(const std::string & str) const;
    std::vector<std::string> split(const std::string & str, char delimiter) const;
};

}

#endif