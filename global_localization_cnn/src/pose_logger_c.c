#include "pose_logger_c.h"
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <geometry_msgs/msg/pose_with_covariance_stamped.h>
#include <std_msgs/msg/header.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#include <libgen.h>

#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)) { return -1; } }
#define RCNOERROR(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)) { fprintf(stderr, "Error: %s\n", #fn); } }

struct PoseLogger {
    rcl_node_t node;
    rcl_subscription_t subscription;
    rcl_allocator_t allocator;
    rcl_context_t context;
    rcl_init_options_t init_options;
    rcl_wait_set_t wait_set;
    
    FILE* log_file;
    char log_file_path[1024];
    char node_name[256];
    
    int messages_count;
    double last_pose_x;
    double last_pose_y;
    double last_pose_z;
    double last_pose_orientation_x;
    double last_pose_orientation_y;
    double last_pose_orientation_z;
    double last_pose_orientation_w;
    double last_pose_covariance[36];
    
    volatile bool running;
    pthread_t thread;
    bool is_thread_running;
    
    pthread_mutex_t mutex;
};

static int create_directory_recursive(const char* path) {
    char temp_path[1024];
    strncpy(temp_path, path, sizeof(temp_path) - 1);
    temp_path[sizeof(temp_path) - 1] = '\0';
    
    char* ptr = temp_path;
    
    if (ptr[0] == '/') {
        ptr++;
    }
    
    char* slash = ptr;
    while ((slash = strchr(slash, '/')) != NULL) {
        *slash = '\0';
        
        if (mkdir(temp_path, 0755) != 0 && errno != EEXIST) {
        }
        
        *slash = '/';
        slash++;
    }
    
    if (mkdir(path, 0755) != 0 && errno != EEXIST) {
        return -1;
    }
    
    return 0;
}

static void generate_log_filename(const char* package_share_path, char* output_path, size_t output_size) {
    time_t now;
    struct tm* timeinfo;
    char timestamp[64];
    
    time(&now);
    timeinfo = localtime(&now);
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", timeinfo);
    
    snprintf(output_path, output_size, "%s/logs", package_share_path);
    
    create_directory_recursive(output_path);
    
    snprintf(output_path, output_size, "%s/logs/pose_log_%s.txt", package_share_path, timestamp);
}

static double quaternion_to_yaw(double x, double y, double z, double w) {
    double siny_cosp = 2.0 * (w * z + x * y);
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    return atan2(siny_cosp, cosy_cosp);
}

static void pose_callback(const void* msg_raw, void* user_data) {
    const geometry_msgs__msg__PoseWithCovarianceStamped* msg = 
        (const geometry_msgs__msg__PoseWithCovarianceStamped*)msg_raw;
    PoseLogger* logger = (PoseLogger*)user_data;
    
    time_t rawtime;
    struct tm* timeinfo;
    char time_buffer[80];
    
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
    
    double yaw = quaternion_to_yaw(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w
    );
    
    char msg_stamp_buffer[80];
    time_t msg_time = msg->header.stamp.sec;
    struct tm* msg_timeinfo = localtime(&msg_time);
    strftime(msg_stamp_buffer, sizeof(msg_stamp_buffer), "%Y-%m-%d %H:%M:%S", msg_timeinfo);
    
    pthread_mutex_lock(&logger->mutex);
    
    logger->messages_count++;
    logger->last_pose_x = msg->pose.pose.position.x;
    logger->last_pose_y = msg->pose.pose.position.y;
    logger->last_pose_z = msg->pose.pose.position.z;
    logger->last_pose_orientation_x = msg->pose.pose.orientation.x;
    logger->last_pose_orientation_y = msg->pose.pose.orientation.y;
    logger->last_pose_orientation_z = msg->pose.pose.orientation.z;
    logger->last_pose_orientation_w = msg->pose.pose.orientation.w;
    
    for (int i = 0; i < 36; i++) {
        logger->last_pose_covariance[i] = msg->pose.covariance[i];
    }
    
    pthread_mutex_unlock(&logger->mutex);
    
    if (logger->log_file) {
        fprintf(logger->log_file, "[%s] POSE #%d:\n", time_buffer, logger->messages_count);
        fprintf(logger->log_file, "  Message timestamp: %s.%u\n", msg_stamp_buffer, msg->header.stamp.nanosec);
        fprintf(logger->log_file, "  Frame: %s\n", msg->header.frame_id.data);
        fprintf(logger->log_file, "  Position: x=%.6f, y=%.6f, z=%.6f\n",
                msg->pose.pose.position.x,
                msg->pose.pose.position.y,
                msg->pose.pose.position.z);
        fprintf(logger->log_file, "  Orientation (quaternion): x=%.6f, y=%.6f, z=%.6f, w=%.6f\n",
                msg->pose.pose.orientation.x,
                msg->pose.pose.orientation.y,
                msg->pose.pose.orientation.z,
                msg->pose.pose.orientation.w);
        fprintf(logger->log_file, "  Yaw: %.6f rad (%.2f deg)\n", yaw, yaw * 180.0 / M_PI);
        
        fprintf(logger->log_file, "  Covariance diagonal: [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
                msg->pose.covariance[0], msg->pose.covariance[7],
                msg->pose.covariance[14], msg->pose.covariance[21],
                msg->pose.covariance[28], msg->pose.covariance[35]);
        fprintf(logger->log_file, "  ---\n");
        
        fflush(logger->log_file);
    }
    
    printf("[C Logger] Received pose #%d: x=%.3f, y=%.3f, yaw=%.2f°\n",
           logger->messages_count,
           msg->pose.pose.position.x,
           msg->pose.pose.position.y,
           yaw * 180.0 / M_PI);
}

static void* run_loop(void* arg) {
    PoseLogger* logger = (PoseLogger*)arg;
    
    while (logger->running) {
        rcl_ret_t ret = rcl_wait(&logger->wait_set, 100000000);
        
        if (ret == RCL_RET_WAIT_TIMEOUT) {
            continue;
        }
        
        if (ret == RCL_RET_OK) {
            if (logger->wait_set.subscriptions[0]) {
                geometry_msgs__msg__PoseWithCovarianceStamped msg;
                geometry_msgs__msg__PoseWithCovarianceStamped__init(&msg);
                
                ret = rcl_take(&logger->subscription, &msg, NULL, NULL);
                if (ret == RCL_RET_OK) {
                    pose_callback(&msg, logger);
                }
                
                geometry_msgs__msg__PoseWithCovarianceStamped__fini(&msg);
            }
        }
        
        rcl_wait_set_fini(&logger->wait_set);
        if (rcl_wait_set_init(&logger->wait_set, 1, 0, 0, 0, 0, 0,
                              &logger->context, logger->allocator) != RCL_RET_OK) {
            fprintf(stderr, "Failed to reinit wait set\n");
            break;
        }
        
        if (rcl_wait_set_add_subscription(&logger->wait_set, &logger->subscription, NULL) != RCL_RET_OK) {
            fprintf(stderr, "Failed to add subscription to wait set\n");
            break;
        }
    }
    
    return NULL;
}

PoseLogger* pose_logger_create(void) {
    PoseLogger* logger = (PoseLogger*)calloc(1, sizeof(PoseLogger));
    if (!logger) return NULL;
    
    pthread_mutex_init(&logger->mutex, NULL);
    logger->running = false;
    logger->is_thread_running = false;
    logger->messages_count = 0;
    logger->last_pose_x = 0.0;
    logger->last_pose_y = 0.0;
    
    return logger;
}

bool pose_logger_init(PoseLogger* logger, const char* node_name, const char* package_share_path) {
    if (!logger) return false;
    
    strncpy(logger->node_name, node_name, sizeof(logger->node_name) - 1);
    
    generate_log_filename(package_share_path, logger->log_file_path, sizeof(logger->log_file_path));
    
    logger->log_file = fopen(logger->log_file_path, "a");
    if (!logger->log_file) {
        fprintf(stderr, "Failed to open log file: %s\n", logger->log_file_path);
        return false;
    }
    
    fprintf(logger->log_file, "========================================\n");
    fprintf(logger->log_file, "=== POSE LOGGER STARTED ===\n");
    fprintf(logger->log_file, "Node: %s\n", node_name);
    fprintf(logger->log_file, "Log file: %s\n", logger->log_file_path);
    time_t now = time(NULL);
    fprintf(logger->log_file, "Start time: %s", ctime(&now));
    fprintf(logger->log_file, "========================================\n\n");
    fflush(logger->log_file);
    
    logger->allocator = rcl_get_default_allocator();
    
    rcl_init_options_init(&logger->init_options, logger->allocator);
    
    rcl_context_init(&logger->context, logger->init_options, logger->allocator);
    
    const char* args[] = {node_name};
    rcl_ret_t ret = rcl_init(1, args, &logger->init_options, &logger->context);
    if (ret != RCL_RET_OK) {
        fprintf(stderr, "Failed to init ROS context\n");
        return false;
    }
    
    rcl_node_options_t node_options = rcl_node_get_default_options();
    ret = rcl_node_init(&logger->node, node_name, "", &logger->context, &node_options);
    if (ret != RCL_RET_OK) {
        fprintf(stderr, "Failed to create node\n");
        return false;
    }
    
    rcl_subscription_options_t sub_options = rcl_subscription_get_default_options();
    
    const rosidl_message_type_support_t* type_support = 
        ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, PoseWithCovarianceStamped);
    
    ret = rcl_subscription_init(&logger->subscription, &logger->node, type_support,
                                "/initialpose", &sub_options);
    if (ret != RCL_RET_OK) {
        fprintf(stderr, "Failed to create subscription to /initialpose\n");
        return false;
    }
    
    ret = rcl_wait_set_init(&logger->wait_set, 1, 0, 0, 0, 0, 0,
                            &logger->context, logger->allocator);
    if (ret != RCL_RET_OK) {
        fprintf(stderr, "Failed to init wait set\n");
        return false;
    }
    
    ret = rcl_wait_set_add_subscription(&logger->wait_set, &logger->subscription, NULL);
    if (ret != RCL_RET_OK) {
        fprintf(stderr, "Failed to add subscription to wait set\n");
        return false;
    }
    
    printf("C Pose Logger initialized. Logging to: %s\n", logger->log_file_path);
    return true;
}

void pose_logger_run(PoseLogger* logger) {
    if (!logger) return;
    
    logger->running = true;
    run_loop(logger);
}

int pose_logger_start_thread(PoseLogger* logger) {
    if (!logger) return -1;
    if (logger->is_thread_running) return -1;
    
    logger->running = true;
    
    if (pthread_create(&logger->thread, NULL, run_loop, logger) != 0) {
        logger->running = false;
        return -1;
    }
    
    logger->is_thread_running = true;
    return 0;
}

void pose_logger_stop(PoseLogger* logger) {
    if (!logger) return;
    
    logger->running = false;
    
    if (logger->is_thread_running) {
        pthread_join(logger->thread, NULL);
        logger->is_thread_running = false;
    }
}

void pose_logger_destroy(PoseLogger* logger) {
    if (!logger) return;
    
    if (logger->log_file) {
        fprintf(logger->log_file, "\n========================================\n");
        fprintf(logger->log_file, "=== POSE LOGGER STOPPED ===\n");
        fprintf(logger->log_file, "Total messages received: %d\n", logger->messages_count);
        time_t now = time(NULL);
        fprintf(logger->log_file, "Stop time: %s", ctime(&now));
        fprintf(logger->log_file, "========================================\n");
        fclose(logger->log_file);
    }
    
    rcl_subscription_fini(&logger->subscription, &logger->node);
    rcl_wait_set_fini(&logger->wait_set);
    rcl_node_fini(&logger->node);
    rcl_context_fini(&logger->context);
    rcl_init_options_fini(&logger->init_options);
    
    pthread_mutex_destroy(&logger->mutex);
    free(logger);
}

int pose_logger_get_messages_count(PoseLogger* logger) {
    if (!logger) return 0;
    pthread_mutex_lock(&logger->mutex);
    int count = logger->messages_count;
    pthread_mutex_unlock(&logger->mutex);
    return count;
}

double pose_logger_get_last_pose_x(PoseLogger* logger) {
    if (!logger) return 0.0;
    pthread_mutex_lock(&logger->mutex);
    double x = logger->last_pose_x;
    pthread_mutex_unlock(&logger->mutex);
    return x;
}

double pose_logger_get_last_pose_y(PoseLogger* logger) {
    if (!logger) return 0.0;
    pthread_mutex_lock(&logger->mutex);
    double y = logger->last_pose_y;
    pthread_mutex_unlock(&logger->mutex);
    return y;
}

double pose_logger_get_last_pose_theta(PoseLogger* logger) {
    if (!logger) return 0.0;
    pthread_mutex_lock(&logger->mutex);
    double theta = quaternion_to_yaw(
        logger->last_pose_orientation_x,
        logger->last_pose_orientation_y,
        logger->last_pose_orientation_z,
        logger->last_pose_orientation_w
    );
    pthread_mutex_unlock(&logger->mutex);
    return theta;
}