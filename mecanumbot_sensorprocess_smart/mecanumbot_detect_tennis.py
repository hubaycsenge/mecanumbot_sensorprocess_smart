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
from std_msgs.msg import String, Int32
from geometry_msgs.msg import PointStamped

import numpy as np
import json
import os

import math
import threading
from concurrent.futures import ThreadPoolExecutor

class TennisBallNode(Node):
    def __init__(self):
        super().__init__('mecanumbot_cam_detect_tennis')
        # Parameters

        self.camera_width = 640.0
        self.camera_fov = math.radians(62.2) # Assume 60 degree horizontal FOV, adjust as needed
        
        self.bridge = CvBridge()
        self.weight_file = 'yolov8n.pt' # Ensure this file is in the 'models' directory of the package
        pkg_share = get_package_share_directory('mecanumbot_sensorprocess_smart')
        self.ball_seen_time = 0.0
        weight_path = os.path.join(pkg_share, 'models', self.weight_file)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        self.get_logger().info(f'Run uses {self.device}')
        self.last_seen_sec = 10000.0
        self.current_time_sec = 0.0
        self.inference_imgsz = 320

        # Keep only one image in flight so the callback cannot build up a backlog.
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.state_lock = threading.Lock()
        self.processing_frame = False

        self.yolo_model = YOLO(weight_path) # Replace with your specific path if needed
        self.yolo_model.to(self.device)
        self.image_subscriber = self.create_subscription(
            CompressedImage,
            'camera/image_raw/compressed',
            self.image_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1))
        self.time_publisher = self.create_publisher(Int32, 'tennis_ball_info', 10)
        self.get_logger().info("TennisBallNode initialized and subscribed to camera/image_raw/compressed")

    def image_callback(self, msg: CompressedImage):
        """Submit image processing task to thread pool"""
        with self.state_lock:
            if self.processing_frame:
                return
            self.processing_frame = True

        self.thread_pool.submit(self._process_image_worker, msg)

    def _process_image_worker(self, msg: CompressedImage):
        """Worker method that processes image in a separate thread"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            results = self.yolo_model.predict(cv_image, imgsz=self.inference_imgsz, verbose=False)

            #32: sports ball
            balls = [x for x in results[0].boxes if int(x.cls[0]) == 32 and float(x.conf[0]) > 0.5]
            
            # Thread-safe update of state
            with self.state_lock:
                if balls:
                    self.last_seen_sec = 0.0
                    self.ball_seen_time = self.get_clock().now().seconds_nanoseconds()[0]
                    self.time_publisher.publish(Int32(data=int(self.last_seen_sec)))
                else:
                    self.current_time_sec = self.get_clock().now().seconds_nanoseconds()[0]
                    self.last_seen_sec = self.current_time_sec - self.ball_seen_time
                    self.time_publisher.publish(Int32(data=int(self.last_seen_sec)))
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
        finally:
            with self.state_lock:
                self.processing_frame = False
        
    def destroy_node(self):
        """Clean up thread pool before destroying node"""
        self.thread_pool.shutdown(wait=True)
        super().destroy_node()
        
def main(args=None):
    rclpy.init(args=args)
    node = TennisBallNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()