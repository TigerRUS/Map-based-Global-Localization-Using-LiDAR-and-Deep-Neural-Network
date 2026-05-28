#!/usr/bin/env python3

import os
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, PointStamped
import math
import time
import tf2_ros
import tf2_geometry_msgs
from rclpy.duration import Duration

class DatasetCollector(Node):
    def __init__(self):
        super().__init__('dataset_collector')
        
        self.dataset_path = os.path.expanduser('~/dataset')
        os.makedirs(self.dataset_path, exist_ok=True)
        
        self.dataset_index = self._get_next_index()
        
        self.map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.map_sub = self.create_subscription(
            OccupancyGrid, 
            '/map', 
            self.map_callback, 
            self.map_qos
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan, 
            '/scan', 
            self.scan_callback, 
            self.scan_qos
        )
        
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, 
            '/amcl_pose', 
            self.odom_callback, 
            self.pose_qos
        )
        
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )
        
        self.map_data = None
        self.map_info = None
        self.scan_data = None
        self.odom_data = None
        
        self.get_logger().info('Dataset Collector initialized')
        self.get_logger().info(f'Dataset path: {self.dataset_path}')
        self.get_logger().info(f'Next dataset index: {self.dataset_index}')
        self.get_logger().info('Press Enter to capture data, Ctrl+C to exit')
    
    def _get_next_index(self):
        index = 1
        while os.path.exists(os.path.join(self.dataset_path, str(index))):
            index += 1
        return index
    
    def map_callback(self, msg):
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
    
    def scan_callback(self, msg):
        self.scan_data = msg
    
    def odom_callback(self, msg):
        self.odom_data = msg
    
    def reset_position(self):
        reset_msg = PoseWithCovarianceStamped()
        reset_msg.header.frame_id = 'map'
        reset_msg.header.stamp = self.get_clock().now().to_msg()
        reset_msg.pose.pose.position.x = 0.0
        reset_msg.pose.pose.position.y = 0.0
        reset_msg.pose.pose.position.z = 0.0
        reset_msg.pose.pose.orientation.x = 0.0
        reset_msg.pose.pose.orientation.y = 0.0
        reset_msg.pose.pose.orientation.z = 0.0
        reset_msg.pose.pose.orientation.w = 1.0
        
        self.initial_pose_pub.publish(reset_msg)
        self.get_logger().info('Position reset to (0,0,0)')
        
        time.sleep(1.0)
    
    def quaternion_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return yaw
    
    def transform_scan_to_map(self):
        if self.map_info is None or self.scan_data is None:
            return None
        
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                self.scan_data.header.frame_id,
                self.scan_data.header.stamp,
                timeout=Duration(seconds=1.0)
            )
            use_transform = True
            self.get_logger().info(f'Transform available: {self.scan_data.header.frame_id} -> map')
        except Exception as e:
            self.get_logger().error(f'Transform error: {e}')
            return None
        
        # Map parameters
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        resolution = self.map_info.resolution
        width = self.map_info.width
        height = self.map_info.height
        
        scan_map = np.zeros((height, width), dtype=np.uint8)
        
        # Process each laser scan beam
        angle_min = self.scan_data.angle_min
        angle_inc = self.scan_data.angle_increment
        points_drawn = 0
        invalid_count = 0
        transform_fails = 0
        
        for i, r in enumerate(self.scan_data.ranges):
            # Check measurement validity
            if r < self.scan_data.range_min or r > self.scan_data.range_max:
                invalid_count += 1
                continue
            
            if math.isnan(r) or math.isinf(r):
                invalid_count += 1
                continue
            
            # Calculate point coordinates in scan frame
            angle = angle_min + i * angle_inc
            px = r * math.cos(angle)
            py = r * math.sin(angle)
            
            # Transform point to map coordinate system
            try:
                point_laser = PointStamped()
                point_laser.header.frame_id = self.scan_data.header.frame_id
                point_laser.header.stamp = self.scan_data.header.stamp
                point_laser.point.x = px
                point_laser.point.y = py
                point_laser.point.z = 0.0
                
                point_map = self.tf_buffer.transform(
                    point_laser, 
                    'map', 
                    timeout=Duration(seconds=0.1)
                )
                px_map = point_map.point.x
                py_map = point_map.point.y
            except Exception as e:
                transform_fails += 1
                continue
            
            # Convert to map pixel coordinates
            u = int((px_map - origin_x) / resolution)
            v = int((py_map - origin_y) / resolution)
            
            # Mark point on map if within bounds
            if 0 <= u < width and 0 <= v < height:
                scan_map[v, u] = 1  # Occupied cell
                points_drawn += 1
        
        self.get_logger().info(f'Points drawn: {points_drawn}, Invalid: {invalid_count}, Transform fails: {transform_fails}, Total: {len(self.scan_data.ranges)}')
        
        if points_drawn == 0:
            self.get_logger().error('No points could be drawn on map!')
            return None
        
        return scan_map
    
    def process_map(self):
        """Convert OccupancyGrid to matrix with proper encoding"""
        if self.map_data is None:
            self.get_logger().error('No map data received')
            return None, None
        
        width = self.map_info.width
        height = self.map_info.height
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        
        processed_map = np.zeros_like(self.map_data, dtype=np.float32)
        processed_map[self.map_data == -1] = 0.5  # unknown
        processed_map[self.map_data == 0] = 0.0   # free
        processed_map[self.map_data == 100] = 1.0  # occupied
        
        map_metadata = {
            'resolution': resolution,
            'origin_x': origin_x,
            'origin_y': origin_y,
            'width': width,
            'height': height
        }
        
        return processed_map, map_metadata
    
    def process_odom(self):
        if self.odom_data is None:
            self.get_logger().error('No odometry data received')
            return None
        
        x = self.odom_data.pose.pose.position.x
        y = self.odom_data.pose.pose.position.y
        
        qx = self.odom_data.pose.pose.orientation.x
        qy = self.odom_data.pose.pose.orientation.y
        qz = self.odom_data.pose.pose.orientation.z
        qw = self.odom_data.pose.pose.orientation.w
        
        self.get_logger().info(f'Raw quaternion: x={qx:.6f}, y={qy:.6f}, z={qz:.6f}, w={qw:.6f}')
        
        yaw = self.quaternion_to_yaw(qx, qy, qz, qw)
        
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)
        
        self.get_logger().info(f'Calculated yaw: {math.degrees(yaw):.2f} degrees, sin={sin_yaw:.4f}, cos={cos_yaw:.4f}')
        
        pose = [x, y, sin_yaw, cos_yaw]
        
        return pose
    
    def save_dataset(self, map_matrix, scan_matrix, pose):
        """Save the dataset to NPZ files"""
        if map_matrix is None or scan_matrix is None or pose is None:
            self.get_logger().error('Cannot save incomplete dataset')
            return False
        
        dataset_dir = os.path.join(self.dataset_path, str(self.dataset_index))
        input_dir = os.path.join(dataset_dir, 'input')
        output_dir = os.path.join(dataset_dir, 'output')
        
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        map_file = os.path.join(input_dir, 'map.npz')
        np.savez_compressed(map_file,
                           data=map_matrix,
                           width=self.map_info.width,
                           height=self.map_info.height,
                           resolution=self.map_info.resolution,
                           origin_x=self.map_info.origin.position.x,
                           origin_y=self.map_info.origin.position.y)
        
        self.get_logger().info(f'Map saved to {map_file}')
        
        scan_file = os.path.join(input_dir, 'scan.npz')
        np.savez_compressed(scan_file,
                           data=scan_matrix,
                           width=self.map_info.width,
                           height=self.map_info.height,
                           resolution=self.map_info.resolution,
                           origin_x=self.map_info.origin.position.x,
                           origin_y=self.map_info.origin.position.y)
        
        self.get_logger().info(f'Scan saved to {scan_file}')
        
        # Save pose as a single line with space-separated values
        pose_file = os.path.join(output_dir, 'pose.txt')
        with open(pose_file, 'w') as f:
            f.write(f"{pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f} {pose[3]:.6f}")
        
        self.get_logger().info(f'Dataset saved to {dataset_dir}')
        self.get_logger().info(f'Pose saved: x={pose[0]:.3f}, y={pose[1]:.3f}, sin={pose[2]:.4f}, cos={pose[3]:.4f}')
        return True
    
    def capture_data(self):
        self.get_logger().info('Capturing data...')
        
        # Step 1: Save current robot pose
        pose = self.process_odom()
        if pose is None:
            return
        
        # Step 2: Reset position to origin
        self.get_logger().info('Resetting position...')
        self.reset_position()
        
        # Step 3: Wait for new data after reset
        self.get_logger().info('Waiting for updated data...')
        time.sleep(2.0)
        
        # Step 4: Transform scan to map coordinates
        self.get_logger().info('Transforming scan to map coordinates...')
        scan_matrix = self.transform_scan_to_map()
        if scan_matrix is None:
            self.get_logger().error('Failed to transform scan')
            return
        
        self.get_logger().info(f'Scan matrix shape: {scan_matrix.shape}')
        
        # Step 5: Process map
        map_matrix, map_metadata = self.process_map()
        if map_matrix is None:
            return
        
        self.get_logger().info(f'Map processed: {map_matrix.shape}')
        
        # Step 6: Save everything
        if self.save_dataset(map_matrix, scan_matrix, pose):
            self.dataset_index += 1
            self.get_logger().info(f'Ready for next capture (index: {self.dataset_index})')

def main(args=None):
    rclpy.init(args=args)
    
    collector = DatasetCollector()
    
    try:
        # Spin in a separate thread
        from threading import Thread
        spin_thread = Thread(target=rclpy.spin, args=(collector,))
        spin_thread.daemon = True
        spin_thread.start()
        
        # Main loop waiting for Enter key
        print("\n" + "="*50)
        print("Dataset Collector for Neural Network")
        print("="*50)
        print(f"Data will be saved to: {os.path.expanduser('~/dataset_np')}")
        print("Press Enter to capture data")
        print("Press Ctrl+C to exit")
        print("="*50 + "\n")
        
        while rclpy.ok():
            try:
                input()  # Wait for Enter key
                collector.capture_data()
            except EOFError:
                break
            except KeyboardInterrupt:
                break
    
    except KeyboardInterrupt:
        pass
    finally:
        collector.destroy_node()
        rclpy.shutdown()
        print("\nDataset collector stopped")

if __name__ == '__main__':
    main()
