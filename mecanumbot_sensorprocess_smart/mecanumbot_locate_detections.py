
import torch
import rclpy
import cv2
from cv_bridge import CvBridge
# ...

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from mecanumbot_msgs.msg import PersonKeypoints, PersonKeypointsArray
from sensor_msgs.msg import CompressedImage, LaserScan
from ament_index_python.packages import get_package_share_directory 
from std_msgs.msg import String
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import Point32, PoseArray, PoseStamped, Pose


import numpy as np
import json
import os

import math

class PersonLocateNode(Node):
    def __init__(self):
        super().__init__('mecanumbot_locate_detections')
        resolved_namespace = self.get_namespace().strip('/')
        self.namespace = resolved_namespace
        self.camera_fov = math.radians(60.0)
        self.robot_frame = f'{self.namespace}/base_footprint' if self.namespace else 'base_footprint'

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher
        self.people_pub = self.create_publisher(String, 'people_coordinates', 10)
        self.transformed_cam_pub = self.create_publisher(PersonKeypointsArray, 'transformed_cam_detections', 10)
        self.cam_people_sub = self.create_subscription(
            PersonKeypointsArray,
            'cam_detected_people',
            self.cam_people_callback,)
        self.laser_people_sub = self.create_subscription(
            PoseArray,
            'lidar_detected_people',
            self.lidar_people_callback,)
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,)
        self.laser_detections = None
        self.cam_detections = None
        self.transformed_cam_detections = PersonKeypointsArray()
        self.scan_data = None
        self.get_logger().info("Person Locate Node has started.")

    def cam_people_callback(self, msg):
        self.cam_detections = msg
        self.transformed_cam_detections = self._transform_cam_detections(msg)
        self.transformed_cam_pub.publish(self.transformed_cam_detections)

    def lidar_people_callback(self, msg):
        self.laser_detections = msg

    def scan_callback(self, msg):
        self.scan_data = msg

    def _get_robot_yaw(self):
        try:
            transform = self.tf_buffer.lookup_transform('map', self.robot_frame, rclpy.time.Time())
            rotation = transform.transform.rotation
            siny_cosp = 2.0 * (rotation.w * rotation.z + rotation.x * rotation.y)
            cosy_cosp = 1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z)
            return math.atan2(siny_cosp, cosy_cosp)
        except Exception:
            return 0.0

    def _keypoint_to_point32(self, keypoint, robot_yaw):
        x_normalized = float(getattr(keypoint, 'x', 0.0))
        y_normalized = float(getattr(keypoint, 'y', 0.0))

        angle_offset = (x_normalized - 0.5) * self.camera_fov
        heading = robot_yaw + angle_offset

        distance = max(0.0, 1.0 - y_normalized)

        point = Point32()
        point.x = distance * math.cos(heading)
        point.y = distance * math.sin(heading)
        point.z = 0.0
        return point


    def merge_detections(self):
        if self.cam_detections is None or self.laser_detections is None:
            return
        
        # Implement your merging logic here, e.g., based on proximity, timestamps, etc.
        # For simplicity, let's just print the number of detections from each source.
        self.get_logger().info(f"Camera detections: {len(self.cam_detections.people)}, Lidar detections: {len(self.laser_detections.poses)}")
        
        # After merging, you can publish the combined results as needed.