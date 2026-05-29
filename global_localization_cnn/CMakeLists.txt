cmake_minimum_required(VERSION 3.8)
project(global_localization_cnn)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS /usr/include/onnxruntime
        /usr/local/include/onnxruntime)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
  HINTS /usr/lib
        /usr/local/lib)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime not found")
endif()

add_library(data_processor SHARED
  src/data_processor.cpp)
target_include_directories(data_processor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(data_processor
  rclcpp::rclcpp
  nav_msgs::nav_msgs
  sensor_msgs::sensor_msgs
  geometry_msgs::geometry_msgs
  tf2::tf2
  tf2_ros::tf2_ros
  tf2_geometry_msgs::tf2_geometry_msgs)

add_executable(global_localization_node
  src/global_localization_node.cpp)
target_include_directories(global_localization_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(global_localization_node
  data_processor
  ${ONNXRUNTIME_LIBRARY})

install(TARGETS data_processor global_localization_node
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME})

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config)

install(FILES
  models/localization_model.onnx
  DESTINATION share/${PROJECT_NAME}/models)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

ament_package()