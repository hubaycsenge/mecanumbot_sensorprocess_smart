# Move these to line 1 and 2
import torch
try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")

# Then import the rest
import rclpy
import cv2
from cv_bridge import CvBridge
# ...

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, LaserScan
from ament_index_python.packages import get_package_share_directory 
from std_msgs.msg import String
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import PointStamped

import numpy as np
import json
import os

import math

# Try importing ultralytics for YOLO


class PersonFusionNode(Node):
    def __init__(self):
        super().__init__('mecanumbot_cam_detect_people')

        # Parameters

        self.camera_width = 640.0
        self.camera_fov = math.radians(62.2) # Assume 60 degree horizontal FOV, adjust as needed
        
        self.bridge = CvBridge()
        self.weight_file = 'yolov8n.pt' # Ensure this file is in the 'models' directory of the package
        pkg_share = get_package_share_directory('mecanumbot_sensorprocess_smart')
        weight_path = os.path.join(pkg_share, 'models', self.weight_file)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo_model = YOLO(weight_path) # Replace with your specific path if needed
        self.yolo_model.to(self.device)
        

        # State variables
        self.latest_scan = None

        # TF2 for frame transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # QoS profile for sensor data (BEST_EFFORT matches lidar & camera publishers)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan', # Assuming standard topic, frame_id: mecanumbot/scan
            self.scan_callback,
            sensor_qos
        )
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed', # frame_id: mecanumbot/head_joint
            self.image_callback,
            sensor_qos
        )

        # Publisher
        self.people_pub = self.create_publisher(String, '/detected_people', 10)

        self.get_logger().info("Person Fusion Node has started. Device: {}".format(self.device))

    def scan_callback(self, msg):
        """Store the latest scan to use when an image arrives."""
        self.latest_scan = msg

    def get_angle_from_x_pixel(self, x_pixel):
        """
        Maps an X pixel coordinate to a horizontal angle (in radians).
        ROS standard: forward is 0, left is positive, right is negative.
        """
        # Center of image is 0 angle. Left of center is positive angle.
        x_offset = (self.camera_width / 2.0) - x_pixel
        angle = x_offset * (self.camera_fov / self.camera_width)
        return angle

    def image_callback(self, msg):
        if self.latest_scan is None:
            self.get_logger().warn("Waiting for scan data...", throttle_duration_sec=2.0)
            return

        # 1. Convert compressed image to OpenCV format
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f"Failed to decode image: {e}")
            return

        # 2. Run YOLO inference
        results = self.yolo_model(cv_image, classes=[0], verbose=False) # class 0 is 'person'
        
        detected_people = []
        scan = self.latest_scan

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 3. Create boundaries on the width axis
                # YOLO returns xyxy (x_min, y_min, x_max, y_max)
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                midline_x = (x_min + x_max) / 2.0

                # 4. Map image width boundaries to LiDAR scan angles
                # x_min is on the left, so it gives a positive (higher) angle
                # x_max is on the right, so it gives a negative (lower) angle
                angle_left = self.get_angle_from_x_pixel(x_min)
                angle_right = self.get_angle_from_x_pixel(x_max)

                # 5. Convert angles to scan indices
                idx_left = int((angle_left - scan.angle_min) / scan.angle_increment)
                idx_right = int((angle_right - scan.angle_min) / scan.angle_increment)

                # Ensure indices are within bounds and order them correctly (min to max)
                idx_start = max(0, min(idx_left, idx_right))
                idx_end = min(len(scan.ranges) - 1, max(idx_left, idx_right))

                # 6. Select points between boundaries and calculate average distance
                if idx_start < idx_end:
                    # Extract the slice of scan ranges
                    target_ranges = scan.ranges[idx_start:idx_end]
                    
                    # Filter out inf and nan values (common in LiDAR data)
                    valid_ranges = [r for r in target_ranges if scan.range_min < r < scan.range_max and not math.isinf(r) and not math.isnan(r)]
                    
                    if valid_ranges:
                        avg_distance = sum(valid_ranges) / len(valid_ranges)
                    else:
                        avg_distance = -1.0 # -1 means no valid LiDAR returns in that bounding box
                else:
                    avg_distance = -1.0

                # Calculate position in robot frame (distance and angle from lidar)
                robot_x = avg_distance * math.cos(angle_left + (angle_right - angle_left) / 2.0)
                robot_y = avg_distance * math.sin(angle_left + (angle_right - angle_left) / 2.0)

                # Try to transform to map frame
                map_position = None
                try:
                    # Create a point in the robot's scan frame
                    point_stamped = PointStamped()
                    point_stamped.header.frame_id = scan.header.frame_id
                    point_stamped.header.stamp = scan.header.stamp
                    point_stamped.point.x = robot_x
                    point_stamped.point.y = robot_y
                    point_stamped.point.z = 0.0

                    # Transform to map frame
                    transformed = self.tf_buffer.transform(point_stamped, 'map', timeout=rclpy.duration.Duration(seconds=0.1))
                    map_position = {
                        'x': float(transformed.point.x),
                        'y': float(transformed.point.y),
                        'z': float(transformed.point.z)
                    }
                except Exception as e:
                    self.get_logger().warn(f"Could not transform to map frame: {e}")

                # Append to our list
                person_data = {
                    'midline_x': float(midline_x),
                    'distance': float(avg_distance),
                    'bounding_box': [float(x_min), float(x_max)],
                    'robot_frame': {
                        'x': float(robot_x),
                        'y': float(robot_y)
                    }
                }
                if map_position:
                    person_data['map_frame'] = map_position
                detected_people.append(person_data)

        # 7. Broadcast the list of people found
        output_msg = String()
        output_msg.data = json.dumps(detected_people)
        self.people_pub.publish(output_msg)

        # Optional: Log the output
        if detected_people:
            self.get_logger().info(f"Published {len(detected_people)} detected people.")

def main(args=None):
    rclpy.init(args=args)
    node = PersonFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()