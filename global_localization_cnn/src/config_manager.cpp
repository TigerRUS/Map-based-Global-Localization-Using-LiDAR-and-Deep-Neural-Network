#include "global_localization_cnn/config_manager.hpp"
#include <iostream>
#include <algorithm>

namespace global_localization_cnn
{

bool ConfigManager::loadFromYAML(const std::string & filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    
    std::string line, current_section;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        
        if (line.find(':') != std::string::npos && 
            line.find('{') == std::string::npos) {
            auto parts = split(line, ':');
            if (parts.size() == 2) {
                std::string key = trim(parts[0]);
                std::string value = trim(parts[1]);
                
                if (key == "model_path") model_config_.model_path = value;
                else if (key == "confidence_threshold") model_config_.confidence_threshold = std::stof(value);
                else if (key == "batch_size") model_config_.batch_size = std::stoi(value);
                else if (key == "scan_topic") sensor_config_.scan_topic = value;
                else if (key == "map_topic") sensor_config_.map_topic = value;
                else if (key == "min_range") sensor_config_.min_range = std::stof(value);
                else if (key == "max_range") sensor_config_.max_range = std::stof(value);
            }
        }
    }
    
    return true;
}

bool ConfigManager::saveToYAML(const std::string & filename) const
{
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    
    file << "# Global Localization Configuration\n\n";
    file << "model:\n";
    file << "  model_path: " << model_config_.model_path << "\n";
    file << "  confidence_threshold: " << model_config_.confidence_threshold << "\n";
    file << "  batch_size: " << model_config_.batch_size << "\n\n";
    file << "sensors:\n";
    file << "  scan_topic: " << sensor_config_.scan_topic << "\n";
    file << "  map_topic: " << sensor_config_.map_topic << "\n";
    file << "  pose_topic: " << sensor_config_.pose_topic << "\n\n";
    file << "filters:\n";
    file << "  enable_map_dilation: " << filter_config_.enable_map_dilation << "\n";
    file << "  dilation_size: " << filter_config_.dilation_size << "\n";
    
    return true;
}

bool ConfigManager::loadFromJSON(const std::string & filename)
{
    (void)filename;
    return false;
}

void ConfigManager::setDefaults()
{
    model_config_ = ModelConfig();
    sensor_config_ = SensorConfig();
    filter_config_ = FilterConfig();
}

std::string ConfigManager::toString() const
{
    std::stringstream ss;
    ss << "ConfigManager:\n"
       << "  Model: " << model_config_.model_path << "\n"
       << "  Confidence: " << model_config_.confidence_threshold << "\n"
       << "  Map Topic: " << sensor_config_.map_topic << "\n"
       << "  Scan Topic: " << sensor_config_.scan_topic;
    return ss.str();
}

std::string ConfigManager::trim(const std::string & str) const
{
    auto start = str.find_first_not_of(" \t\n\r");
    auto end = str.find_last_not_of(" \t\n\r");
    return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

std::vector<std::string> ConfigManager::split(const std::string & str, char delimiter) const
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(trim(token));
    }
    return tokens;
}

std::string ConfigManager::escapeYAML(const std::string & str) const
{
    return "\"" + str + "\"";
}

}