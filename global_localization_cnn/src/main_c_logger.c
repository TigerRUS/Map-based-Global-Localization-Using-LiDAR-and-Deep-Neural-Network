#include "pose_logger_c.h"
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

static PoseLogger* g_logger = NULL;

void signal_handler(int sig) {
    printf("\nShutting down pose logger...\n");
    if (g_logger) {
        pose_logger_stop(g_logger);
        pose_logger_destroy(g_logger);
    }
    exit(0);
}

static void get_package_share_path(char* output, size_t output_size) {
    const char* ament_prefix = getenv("AMENT_PREFIX_PATH");
    if (ament_prefix) {
        const char* colon = strchr(ament_prefix, ':');
        size_t len = colon ? (colon - ament_prefix) : strlen(ament_prefix);
        if (len < output_size - 32) {
            strncpy(output, ament_prefix, len);
            output[len] = '\0';
            snprintf(output + len, output_size - len, "/share/global_localization_cnn");
            return;
        }
    }
    
    getcwd(output, output_size);
    strncat(output, "/install/global_localization_cnn/share/global_localization_cnn", 
            output_size - strlen(output) - 1);
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    g_logger = pose_logger_create();
    if (!g_logger) {
        fprintf(stderr, "Failed to create logger\n");
        return -1;
    }
    
    char package_share_path[1024];
    if (argc > 1) {
        strncpy(package_share_path, argv[1], sizeof(package_share_path) - 1);
    } else {
        get_package_share_path(package_share_path, sizeof(package_share_path));
    }
    
    printf("Package share path: %s\n", package_share_path);
    
    if (!pose_logger_init(g_logger, "pose_logger_c", package_share_path)) {
        fprintf(stderr, "Failed to initialize logger\n");
        pose_logger_destroy(g_logger);
        return -1;
    }
    
    printf("Starting C pose logger. Logs will be saved to: %s/logs/\n", package_share_path);
    printf("Press Ctrl+C to stop\n\n");
    
    if (pose_logger_start_thread(g_logger) != 0) {
        fprintf(stderr, "Failed to start logger thread\n");
        pose_logger_destroy(g_logger);
        return -1;
    }
    
    while (1) {
        sleep(10);
        int count = pose_logger_get_messages_count(g_logger);
        if (count > 0) {
            printf("[Status] Received %d poses. Last pose: (%.3f, %.3f, %.3f rad)\n",
                   count,
                   pose_logger_get_last_pose_x(g_logger),
                   pose_logger_get_last_pose_y(g_logger),
                   pose_logger_get_last_pose_theta(g_logger));
        }
    }
    
    return 0;
}