
import torch
try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")

import rclpy
import cv2
from cv_bridge import CvBridge
import numpy as np
# ...

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy,DurabilityPolicy
from sensor_msgs.msg import LaserScan
from mecanumbot_msgs.msg import CamPersonDetectionArray,CamPersonDetection
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import CompressedImage
from ament_index_python.packages import get_package_share_directory 
from std_msgs.msg import String
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import Pose

import numpy as np
import transforms3d as t3d
import json
import os

import math




class PersonDetectNode(Node):
    def __init__(self,namespace=''):
        super().__init__('mecanumbot_cam_detect_people')
        self.declare_parameters(
        namespace=namespace,
        parameters=[
        ('camera_params.camera_width', 640.0),
        ('camera_params.camera_height', 480.0),
        ('camera_params.camera_fov', math.radians(60.0)),
        ('img_process_params.weight_file', 'yolo26n-pose.pt')
         ])

        # Parameters

        self.camera_width = self.get_parameter('camera_params.camera_width').value
        self.camera_fov = self.get_parameter('camera_params.camera_fov').value # Assume 60 degree horizontal FOV, adjust as needed

        self.bridge = CvBridge()
        self.weight_file = self.get_parameter('img_process_params.weight_file').value # Ensure this file is in the 'models' directory of the package
        pkg_share = get_package_share_directory('mecanumbot_sensorprocess_smart')
        resolved_namespace = self.get_namespace().strip('/')
        self.namespace = resolved_namespace
        #weight_path = os.path.join(pkg_share, 'models', self.weight_file)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo_model = YOLO(self.weight_file) # Replace with your specific path if needed
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
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST
        )

        self.image_sub = self.create_subscription(
            CompressedImage,
            'camera/image_raw/compressed', # frame_id: mecanumbot/head_joint
            self.image_callback,
            sensor_qos
        )

        self.robot_subscriber =self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self.amcl_callback,
            qos_profile
        )
        # Publisher
        self.people_pub = self.create_publisher(CamPersonDetectionArray, 'cam_people_detections', 10)
        self.get_logger().info("Person Detect Node has started. Device: {}".format(self.device))
        

    def amcl_callback(self, msg):
        self.robot_pose = msg.pose.pose
        self.robot_orientation_quat = self.robot_pose.orientation
        self.robot_orientation_euler = t3d.euler.quat2euler([
            self.robot_orientation_quat.x,
            self.robot_orientation_quat.y,
            self.robot_orientation_quat.z,
            self.robot_orientation_quat.w
        ])

        self.camera_left_yaw = self.robot_orientation_euler[2] + self.camera_fov / 2
        self.camera_right_yaw = self.robot_orientation_euler[2] - self.camera_fov / 2

    
    def XYN_to_Pose(self, xyn):
        msg = Pose()
        msg.position.x = xyn[0]
        msg.position.y = xyn[1]
        msg.position.z = 0.0
        return msg
    
    def cam_to_angle(self,X):
        X_inv = 1 - X # camera pixel indexing direction is opposite of robot frame dir
        angle = (1-X_inv) * self.camera_right_yaw + X_inv * self.camera_left_yaw
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
        detected_people_bounds = []
        for result in results:
            xyn = result.keypoints.xyn # Normalized keypoints (x, y in [0,1])
            
            person_msg = CamPersonDetection()
            person_msg.keypoints.nose = self.XYN_to_Pose(xyn[0]) # Nose keypoint
            person_msg.keypoints.left_eye = self.XYN_to_Pose(xyn[1]) # Left eye keypoint
            person_msg.keypoints.right_eye = self.XYN_to_Pose(xyn[2]) # Right eye keypoint
            person_msg.keypoints.left_ear = self.XYN_to_Pose(xyn[3]) # Left ear keypoint
            person_msg.keypoints.right_ear = self.XYN_to_Pose(xyn[4]) # Right ear keypoint
            person_msg.keypoints.left_shoulder = self.XYN_to_Pose(xyn[5]) # Left shoulder keypoint
            person_msg.keypoints.right_shoulder = self.XYN_to_Pose(xyn[6]) # Right shoulder keypoint
            person_msg.keypoints.left_elbow = self.XYN_to_Pose(xyn[7]) # Left elbow keypoint
            person_msg.keypoints.right_elbow = self.XYN_to_Pose(xyn[8]) # Right elbow keypoint
            person_msg.keypoints.left_wrist = self.XYN_to_Pose(xyn[9]) # Left wrist keypoint
            person_msg.keypoints.right_wrist = self.XYN_to_Pose(xyn[10]) # Right wrist keypoint
            person_msg.keypoints.left_hip = self.XYN_to_Pose(xyn[11]) # Left hip keypoint
            person_msg.keypoints.right_hip = self.XYN_to_Pose(xyn[12]) # Right hip keypoint
            person_msg.keypoints.left_knee = self.XYN_to_Pose(xyn[13]) # Left knee keypoint
            person_msg.keypoints.right_knee = self.XYN_to_Pose(xyn[14]) # Right knee keypoint
            person_msg.keypoints.left_ankle = self.XYN_to_Pose(xyn[15]) # Left ankle keypoint
            person_msg.keypoints.right_ankle = self.XYN_to_Pose(xyn[16]) # Right ankle keypoint
            xyn_X = np.array(xyn)[:,0]
            X_max, X_min = xyn_X.min(), xyn_X.max() # camera pixel indexing direction is opposite of robot frame dir, so max X is leftmost point and min X is rightmost point
            X_min_angle = self.cam_to_angle(X_min)
            X_max_angle = self.cam_to_angle(X_max)
            person_msg.bound_angle_min = X_min_angle
            person_msg.bound_angle_max = X_max_angle


        out_msg = CamPersonDetectionArray()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = f'{self.namespace}/head_link'
        out_msg.people = detected_people
        self.people_raw_pub.publish(out_msg)
        #self.people_pub.publish(out_msg)
        #self.get_logger().info(f"Published {len(detected_people)} detected people.")

            

def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()