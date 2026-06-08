
import torch
import rclpy
import cv2
from cv_bridge import CvBridge
# ...

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from mecanumbot_msgs.msg import CamPersonDetectionArray
from sensor_msgs.msg import CompressedImage, LaserScan
from ament_index_python.packages import get_package_share_directory 
from std_msgs.msg import String
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
from geometry_msgs.msg import Point32, PoseArray, PoseWithCovarianceStamped, Pose


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

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST
        )
        
        # Publisher
        self.people_pub = self.create_publisher(PoseArray, 'people_fusion', 10)
        self.cam_people_sub = self.create_subscription(
            CamPersonDetectionArray,
            'cam_detected_people',
            self.cam_people_callback,)
        self.robot_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_pose',
            self.robot_pose_callback,)
        self.laser_people_sub = self.create_subscription(
            PoseArray,
            'lidar_detected_people',
            self.lidar_people_callback,)
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,)
        self.laser_detections = None
        self.laser_angles
        self.cam_detections = None
        self.scan_data = None
        self.get_logger().info("Person Locate Node has started.")

    def cam_people_callback(self, msg):
        self.cam_detections = msg.people

    def lidar_people_callback(self, msg):
        transform = self.tf_buffer.lookup_transform(
                                                    'mecanumbot/base_link',
                                                    'mecanumbot/base_scan',
                                                    rclpy.time.Time(),  
                                                )
        self.laser_detections = [tf2_geometry_msgs.do_transform_pose(pose, transform) for pose in msg.poses]
        self.laser_angles = {pose:math.atan2(pose.position.y, pose.position.x) for pose in self.laser_detections}

    def scan_callback(self, msg):
        self.scan_data = msg

    def arrange_with_scan_dets(self,person):
        pose_candidates = []
        for laser_pose, angle in self.laser_angles.items():
            if person.bound_angle_min <= angle <= person.bound_angle_max:
                # This laser detection is within the camera detection's angular bounds
                # You can further check distance, timestamp, etc. to confirm the match
                pose_candidates.append([laser_pose])
                self.get_logger().info(f"Camera detection at angle {angle} matches laser detection at position ({laser_pose.position.x}, {laser_pose.position.y})")
                # Here you can create a new FusionPersonDetection message combining data from both sources
                # and publish it as needed
        return pose_candidates[0] if pose_candidates else None #TODO: algorithm to select the best candidate if multiple matches are found
    def extrap_from_raw_scan(self, person):
        if self.scan_data is None:
            return None
        
        min_scan_index = int((person.bound_angle_min - self.scan_data.angle_min) / self.scan_data.angle_increment)
        max_scan_index = int((person.bound_angle_max - self.scan_data.angle_min) / self.scan_data.angle_increment)
        angle = person.bound_angle_min + (person.bound_angle_max - person.bound_angle_min) / 2
        distances = self.scan_data.ranges[min_scan_index:max_scan_index+1]
        dist_avg = sum(distances) / len(distances) if distances else None
        if not distances:
            return None
            distance = self.scan_data.ranges[scan_index]
        if dist_avg is not None and dist_avg < self.scan_data.range_max:
            # Convert polar coordinates (distance, angle) to Cartesian coordinates (x, y)
            x = dist_avg * math.cos(angle)
            y = dist_avg * math.sin(angle)
            self.get_logger().info(f"Extrapolated position for camera detection: ({x}, {y}) at angle {angle}")
            return Pose(position=Point32(x=x, y=y, z=0.0))
        
        return None
    def merge_detections(self):
        if self.cam_detections is None:
            return
        
        # Implement your merging logic here, e.g., based on proximity, timestamps, etc.
        # For simplicity, let's just print the number of detections from each source.
        self.get_logger().info(f"Camera detections: {len(self.cam_detections.people)}")
        for person in self.cam_detections:
            person_pose = self.arrange_with_scan_dets(person)    
            if person_pose is None:
                person_pose = self.extrap_from_raw_scan(person)
                self.get_logger().info(f"Extrapolated pose for person: ({person_pose.position.x}, {person_pose.position.y})") if person_pose else self.get_logger().info("Could not extrapolate pose for person.")
            
        # After merging, you can publish the combined results as needed.

    def timer_callback(self):
        self.merge_detections()