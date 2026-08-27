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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan
from mecanumbot_msgs.msg import CamPersonDetectionArray, CamPersonDetection
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import CompressedImage
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import String, Float32
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import Pose

import numpy as np
import transforms3d as t3d
import json
import os

import math


class PersonDetectNode(Node):
    def __init__(self, namespace=""):
        super().__init__("mecanumbot_cam_detect_people")
        self.declare_parameters(
            namespace=namespace,
            parameters=[
                ("camera_params.camera_width", 640.0),
                ("camera_params.camera_height", 480.0),
                ("camera_params.camera_fov", math.radians(60.0)),
                ("from_topic", False),
                ("camera_topic", "camera/image_raw/compressed"),
                ("webcam_device", "/dev/video0"),
                ("img_process_params.weight_file", "yolo26n-pose.pt"),
            ],
        )

        # Parameters

        self.camera_width = self.get_parameter("camera_params.camera_width").value
        self.camera_height = self.get_parameter("camera_params.camera_height").value
        self.camera_fov = self.get_parameter(
            "camera_params.camera_fov"
        ).value  # Assume 60 degree horizontal FOV, adjust as needed
        self.from_topic = self.get_parameter("from_topic").value
        self.camera_topic = self.get_parameter("camera_topic").value
        self.webcam_device = self.get_parameter("webcam_device").value

        self.bridge = CvBridge()
        self.weight_file = self.get_parameter(
            "img_process_params.weight_file"
        ).value  # Ensure this file is in the 'models' directory of the package
        pkg_share = get_package_share_directory("mecanumbot_sensorprocess_smart")
        weight_path = os.path.join(pkg_share, "models", self.weight_file)
        resolved_namespace = self.get_namespace().strip("/")
        self.namespace = resolved_namespace
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_model = YOLO(
            weight_path
        )  # Load pose weights from the package share directory
        self.yolo_model.to(self.device)
        self.detected_people = CamPersonDetectionArray()
        self.webcam_capture = None
        self.webcam_timer = None
        self.robot_pose = None

        # TF2 for frame transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # QoS profile for sensor data (BEST_EFFORT matches lidar & camera publishers)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )

        if self.from_topic:
            self.image_sub = self.create_subscription(
                CompressedImage,
                self.camera_topic,  # frame_id: mecanumbot/head_joint
                self.image_callback,
                sensor_qos,
            )
            self.get_logger().info(f"Using camera topic input: {self.camera_topic}")
        else:
            self.webcam_capture = cv2.VideoCapture(self.webcam_device, cv2.CAP_V4L2)
            self.webcam_capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.camera_width))
            self.webcam_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.camera_height))
            if not self.webcam_capture.isOpened():
                self.get_logger().error(
                    f"Failed to open webcam device: {self.webcam_device}"
                )
            else:
                self.webcam_timer = self.create_timer(1.0 / 15.0, self.webcam_callback)
                self.get_logger().info(
                    f"Using webcam input from {self.webcam_device} at {int(self.camera_width)}x{int(self.camera_height)}"
                )

        self.robot_subscriber = self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self.amcl_callback, qos_profile
        )
        # Publisher
        self.people_pub = self.create_publisher(
            CamPersonDetectionArray, "cam_people_detections", 10
        )
        self.get_logger().info(
            "Person Detect Node has started. Device: {}".format(self.device)
        )
        self.get_logger().info(
            f"Camera source mode: {'topic' if self.from_topic else 'webcam'}"
        )

    def amcl_callback(self, msg):
        self.get_logger().debug(f"Received AMCL pose update: {msg.pose.pose}")
        self.robot_pose = msg.pose.pose
        self.robot_orientation_quat = self.robot_pose.orientation
        self.robot_orientation_euler = t3d.euler.quat2euler(
            [
                self.robot_orientation_quat.w,
                self.robot_orientation_quat.x,
                self.robot_orientation_quat.y,
                self.robot_orientation_quat.z,
            ]
        )

        self.camera_left_yaw = self.robot_orientation_euler[2] + self.camera_fov / 2
        self.camera_right_yaw = self.robot_orientation_euler[2] - self.camera_fov / 2

    def XYN_to_Pose(self, xyn):
        msg = Pose()
        msg.position.x = float(xyn[0])
        msg.position.y = float(xyn[1])
        msg.position.z = 0.0
        return msg

    def cam_to_angle(self, X):
        X_inv = 1 - X  # camera pixel indexing direction is opposite of robot frame dir
        angle = (1 - X_inv) * self.camera_right_yaw + X_inv * self.camera_left_yaw
        self.get_logger().info(
            f"####### Calculated angle: {angle} from X: {X} with camera FOV: {math.degrees(self.camera_fov)} degrees"
        )
        return angle

    def process_image(self, cv_image):
        results = self.yolo_model(
            cv_image, classes=[0], verbose=False
        )  # class 0 is 'person'
        detected_people = []
        if len(results) > 0:
            for result in results:
                try:
                    xyn = result.keypoints.xyn.cpu().numpy()[
                        0, :, :
                    ] 
                except Exception as e:
                    self.get_logger().error(f"Failed to process keypoints: {e}")
                    continue
                self.get_logger().info(f"Received shape: {xyn.shape}")
                if len(xyn) != 17:
                    self.get_logger().warn(
                        f"Expected 17 keypoints, got {len(xyn)}. Skipping detection."
                    )
                    self.get_logger().info(f"Keypoints data: {xyn}")
                    continue
                person_msg = CamPersonDetection()
                person_msg.keypoints.nose = self.XYN_to_Pose(xyn[0, :])
                person_msg.keypoints.left_eye = self.XYN_to_Pose(xyn[1, :])
                person_msg.keypoints.right_eye = self.XYN_to_Pose(xyn[2, :])
                person_msg.keypoints.left_ear = self.XYN_to_Pose(xyn[3, :])
                person_msg.keypoints.right_ear = self.XYN_to_Pose(xyn[4, :])
                person_msg.keypoints.left_shoulder = self.XYN_to_Pose(xyn[5, :])
                person_msg.keypoints.right_shoulder = self.XYN_to_Pose(xyn[6, :])
                person_msg.keypoints.left_elbow = self.XYN_to_Pose(xyn[7, :])
                person_msg.keypoints.right_elbow = self.XYN_to_Pose(xyn[8, :])
                person_msg.keypoints.left_wrist = self.XYN_to_Pose(xyn[9, :])
                person_msg.keypoints.right_wrist = self.XYN_to_Pose(xyn[10, :])
                person_msg.keypoints.left_hip = self.XYN_to_Pose(xyn[11, :])
                person_msg.keypoints.right_hip = self.XYN_to_Pose(xyn[12, :])
                person_msg.keypoints.left_knee = self.XYN_to_Pose(xyn[13, :])
                person_msg.keypoints.right_knee = self.XYN_to_Pose(xyn[14, :])
                person_msg.keypoints.left_ankle = self.XYN_to_Pose(xyn[15, :])
                person_msg.keypoints.right_ankle = self.XYN_to_Pose(xyn[16, :])
                xyn_X = np.array(xyn)[:, 0]
                X_max, X_min = (
                    xyn_X.min(),
                    xyn_X.max(),
                )  # camera pixel indexing direction is opposite of robot frame dir, so max X is leftmost point and min X is rightmost point
                if self.robot_pose is not None:
                    X_min_angle = self.cam_to_angle(X_min)
                    X_max_angle = self.cam_to_angle(X_max)
                    person_msg.bound_angle_min = Float32(data=X_min_angle)
                    person_msg.bound_angle_max = Float32(data=X_max_angle)
                else:
                    self.get_logger().warn(
                        "Robot pose is None, cannot calculate bound angles."
                    )
                detected_people.append(person_msg)

        self.detected_people.header.stamp = self.get_clock().now().to_msg()
        self.detected_people.header.frame_id = f"{self.namespace}/head_link"
        self.detected_people.people = detected_people
        self.people_pub.publish(self.detected_people)
        self.get_logger().info(f"Published {len(detected_people)} detected people.")

    def image_callback(self, msg):
        self.get_logger().info(
            f"Received image message with size: {len(msg.data)} bytes"
        )
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        except Exception as e:
            self.get_logger().error(f"Failed to decode image: {e}")
            return
        if self.robot_pose is not None:
            self.process_image(cv_image)

    def webcam_callback(self):
        if self.webcam_capture is None or not self.webcam_capture.isOpened():
            return

        ok, frame = self.webcam_capture.read()
        if not ok:
            self.get_logger().warn(
                f"Failed to read frame from webcam device: {self.webcam_device}"
            )
            return

        self.process_image(frame)

    def destroy_node(self):
        if self.webcam_timer is not None:
            self.webcam_timer.cancel()
            self.webcam_timer = None
        if self.webcam_capture is not None:
            self.webcam_capture.release()
            self.webcam_capture = None
        super().destroy_node()


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


if __name__ == "__main__":
    main()
