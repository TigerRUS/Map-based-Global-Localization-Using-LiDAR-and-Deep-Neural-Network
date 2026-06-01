#ifndef POSE_LOGGER_C_H
#define POSE_LOGGER_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

typedef struct PoseLogger PoseLogger;

PoseLogger* pose_logger_create(void);

bool pose_logger_init(PoseLogger* logger, const char* node_name, const char* package_share_path);

void pose_logger_run(PoseLogger* logger);

int pose_logger_start_thread(PoseLogger* logger);

void pose_logger_stop(PoseLogger* logger);
void pose_logger_destroy(PoseLogger* logger);

int pose_logger_get_messages_count(PoseLogger* logger);
double pose_logger_get_last_pose_x(PoseLogger* logger);
double pose_logger_get_last_pose_y(PoseLogger* logger);
double pose_logger_get_last_pose_theta(PoseLogger* logger);

#ifdef __cplusplus
}
#endif

#endif