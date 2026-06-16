import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from mecanumbot_msgs.msg import CamPersonDetectionArray, CamPersonDetection
from geometry_msgs.msg import Pose
import os
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib
import pyds
import ctypes
import numpy as np
import cv2
import math
import transforms3d as t3d
from ament_index_python.packages import get_package_share_directory


class DeepStreamPersonDetectNode(Node):
    def __init__(self, namespace=''):
        super().__init__('mecanumbot_cam_detect_people_ds')
        self.declare_parameters(
            namespace=namespace,
            parameters=[
                ('camera_params.camera_width', 640),
                ('camera_params.camera_height', 480),
                ('camera_params.camera_fov', math.radians(60.0)),
                ('from_topic', False),
                ('camera_topic', 'camera/image_raw/compressed'),
                ('webcam_device', '/dev/video0')
            ]
        )

        self.camera_width = self.get_parameter('camera_params.camera_width').value
        self.camera_height = self.get_parameter('camera_params.camera_height').value
        self.from_topic = self.get_parameter('from_topic').value
        self.webcam_device = self.get_parameter('webcam_device').value

        # Initialize GStreamer
        Gst.init(None)
        self.pipeline = Gst.Pipeline()

        # Build Pipeline Elements
        self.get_logger().info('source: ' + str(self.from_topic))
        if self.from_topic:
            self.source = Gst.ElementFactory.make("appsrc", "ros-image-source")
            self.source.set_property("is-live", True)
            caps = Gst.Caps.from_string(f"video/x-raw, format=BGR, width={self.camera_width}, height={self.camera_height}, framerate=15/1")
            self.source.set_property("caps", caps)
            
            # ROS Subscriber to feed appsrc
            sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
            self.image_sub = self.create_subscription(CompressedImage, self.get_parameter('camera_topic').value, self.image_callback, sensor_qos)
        else:
            self.source = Gst.ElementFactory.make("v4l2src", "webcam-source")
            self.source.set_property("device", self.webcam_device)
            
        self.vidconv_src = Gst.ElementFactory.make("nvvideoconvert", "convertor_src")
        self.mux = Gst.ElementFactory.make("nvstreammux", "muxer")
        self.mux.set_property("width", self.camera_width)
        self.mux.set_property("height", self.camera_height)
        self.mux.set_property("batch-size", 1)
        self.mux.set_property("batched-push-timeout", 40000)

        self.nvinfer = Gst.ElementFactory.make("nvinfer", "primary-inference")
        # POINT THIS TO YOUR CONFIG FILE
        path = get_package_share_directory('mecanumbot_sensorprocess_smart')
        self.nvinfer.set_property("config-file-path", os.path.join(path, 'deepstream_config', 'config_infer_yolo26_pose.txt'))

        self.sink = Gst.ElementFactory.make("fakesink", "fakesink")

        # Add elements to pipeline
        for elem in [self.source, self.vidconv_src, self.mux, self.nvinfer, self.sink]:
            self.pipeline.add(elem)

        # Link elements
        if self.from_topic:
            self.source.link(self.vidconv_src)
        else:
            self.source.link(self.vidconv_src)
            
        vidconv_src_pad = self.vidconv_src.get_static_pad("src")
        mux_sink_pad = self.mux.get_request_pad("sink_0")
        vidconv_src_pad.link(mux_sink_pad)
        
        self.mux.link(self.nvinfer)
        self.nvinfer.link(self.sink)

        # Attach Probe to extract metadata from the GPU
        infer_src_pad = self.nvinfer.get_static_pad("src")
        infer_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.metadata_probe, 0)

        # Publisher
        self.people_pub = self.create_publisher(CamPersonDetectionArray, 'cam_people_detections', 10)

        # Start Pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        self.get_logger().info("DeepStream Pipeline Running!")

    def image_callback(self, msg):
        """Pushes ROS images into the DeepStream Pipeline."""
        self.get_logger().debug("Received image from ROS topic.")
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            data = cv_image.tobytes()
            buffer = Gst.Buffer.new_allocate(None, len(data), None)
            buffer.fill(0, data)
            self.source.emit("push-buffer", buffer)
        except Exception as e:
            self.get_logger().error(f"Image decode failed: {e}")

    def XYN_to_Pose(self, x, y):
        self.get_logger().debug(f"Converting normalized coordinates ({x}, {y}) to Pose.")
        msg = Pose()
        msg.position.x = float(x)
        msg.position.y = float(y)
        msg.position.z = 0.0
        return msg

    def metadata_probe(self, pad, info, u_data):
        """Runs asynchronously when the GPU finishes inference on a frame."""
        self.get_logger().info(">>> Probe triggered! Data is flowing through the network.", throttle_duration_sec=2.0)
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        
        detected_people = []

        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                
                self.get_logger().info(f"Object class_id: {obj_meta.class_id}, confidence: {obj_meta.confidence}")
                self.get_logger().info(f"Object bounding box: ({obj_meta.rect_params.left}, {obj_meta.rect_params.top}, {obj_meta.rect_params.width}, {obj_meta.rect_params.height})")
                self.get_logger().info(f"Object mask_params size: {obj_meta.mask_params.data}")
                #self.get_logger().info(f"Object rect_params data: {obj_meta.rect_params.left}, {obj_meta.rect_params.top}, {obj_meta.rect_params.width}, {obj_meta.rect_params.height}")
                self.get_logger().info(f'object text params: {obj_meta.text_params.display_text}')

                # --- THE FIX: EXTRACT KEYPOINTS FROM MASK_PARAMS ---
                mask_params = obj_meta.mask_params
                
                # Check if the C++ parser successfully injected data into the mask array
                if mask_params.size > 0:
                    self.get_logger().info(f">>> Keypoints found in mask_params!")
                    
                    # get_mask_array() natively returns a NumPy array!
                    # We just flatten it (if it isn't already) and reshape it back into our 17x3 matrix
                    raw_data = mask_params.get_mask_array()
                    keypoints = np.array(raw_data).flatten().reshape((17, 3))
                    
                    person_msg = CamPersonDetection()
                    
                    # NOTE: DeepStream outputs keypoints in absolute StreamMux coordinates (e.g. 640x480).
                    # Your division successfully normalizes them back to 0.0 - 1.0 (XYN) format!
                    person_msg.keypoints.nose = self.XYN_to_Pose(keypoints[0][0] / self.camera_width, keypoints[0][1] / self.camera_height)
                    person_msg.keypoints.left_eye = self.XYN_to_Pose(keypoints[1][0] / self.camera_width, keypoints[1][1] / self.camera_height)
                    person_msg.keypoints.right_eye = self.XYN_to_Pose(keypoints[2][0] / self.camera_width, keypoints[2][1] / self.camera_height)
                    person_msg.keypoints.left_ear = self.XYN_to_Pose(keypoints[3][0] / self.camera_width, keypoints[3][1] / self.camera_height)
                    person_msg.keypoints.right_ear = self.XYN_to_Pose(keypoints[4][0] / self.camera_width, keypoints[4][1] / self.camera_height)
                    person_msg.keypoints.left_shoulder = self.XYN_to_Pose(keypoints[5][0] / self.camera_width, keypoints[5][1] / self.camera_height)
                    person_msg.keypoints.right_shoulder = self.XYN_to_Pose(keypoints[6][0] / self.camera_width, keypoints[6][1] / self.camera_height)
                    person_msg.keypoints.left_elbow = self.XYN_to_Pose(keypoints[7][0] / self.camera_width, keypoints[7][1] / self.camera_height)
                    person_msg.keypoints.right_elbow = self.XYN_to_Pose(keypoints[8][0] / self.camera_width, keypoints[8][1] / self.camera_height)
                    person_msg.keypoints.left_wrist = self.XYN_to_Pose(keypoints[9][0] / self.camera_width, keypoints[9][1] / self.camera_height)
                    person_msg.keypoints.right_wrist = self.XYN_to_Pose(keypoints[10][0] / self.camera_width, keypoints[10][1] / self.camera_height)
                    person_msg.keypoints.left_hip = self.XYN_to_Pose(keypoints[11][0] / self.camera_width, keypoints[11][1] / self.camera_height)
                    person_msg.keypoints.right_hip = self.XYN_to_Pose(keypoints[12][0] / self.camera_width, keypoints[12][1] / self.camera_height)
                    person_msg.keypoints.left_knee = self.XYN_to_Pose(keypoints[13][0] / self.camera_width, keypoints[13][1] / self.camera_height)
                    person_msg.keypoints.right_knee = self.XYN_to_Pose(keypoints[14][0] / self.camera_width, keypoints[14][1] / self.camera_height)
                    person_msg.keypoints.left_ankle = self.XYN_to_Pose(keypoints[15][0] / self.camera_width, keypoints[15][1] / self.camera_height)
                    person_msg.keypoints.right_ankle = self.XYN_to_Pose(keypoints[16][0] / self.camera_width, keypoints[16][1] / self.camera_height)
                    
                    # (Remember to port over your bound_angle_min/max calculations here if you still need them!)
                    
                    detected_people.append(person_msg)
                else:
                    self.get_logger().warn("Object detected, but no keypoint mask_params were found!")

                l_obj = l_obj.next
        # Publish back to ROS
        if detected_people:
            self.get_logger().info(f"Publishing {len(detected_people)} detected people.")
            msg_array = CamPersonDetectionArray()
            msg_array.header.stamp = self.get_clock().now().to_msg()
            msg_array.people = detected_people
            self.people_pub.publish(msg_array)

        return Gst.PadProbeReturn.OK

def main(args=None):
    rclpy.init(args=args)
    node = DeepStreamPersonDetectNode()
    rclpy.spin(node)
    node.pipeline.set_state(Gst.State.NULL)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()