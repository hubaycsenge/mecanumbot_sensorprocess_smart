import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from mecanumbot_msgs.msg import CamPersonDetectionArray, CamPersonDetection
from std_msgs.msg import Float32 as Float
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

# Standard YOLO pose skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),                 # Head/Face
    (5, 6),                                         # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),                # Arms
    (11, 12), (5, 11), (6, 12),                     # Torso/Hips
    (11, 13), (13, 15), (12, 14), (14, 16)          # Legs
]

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
                ('webcam_device', '/dev/video0'),
                ('debug_mode', False)
            ]
        )

        self.camera_width = self.get_parameter('camera_params.camera_width').value
        self.camera_height = self.get_parameter('camera_params.camera_height').value
        self.camera_fov = self.get_parameter('camera_params.camera_fov').value
        self.from_topic = self.get_parameter('from_topic').value
        self.webcam_device = self.get_parameter('webcam_device').value
        self.Y_padding = (self.camera_width - self.camera_height)/2
        self.debug_mode = self.get_parameter('debug_mode').value

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
        path = get_package_share_directory('mecanumbot_sensorprocess_smart')
        self.nvinfer.set_property("config-file-path", os.path.join(path, 'deepstream_config', 'config_infer_yolo26_pose.txt'))

        # --- NEW ELEMENTS FOR IMAGE EXTRACTION ---
        # Converts infer output format to RGBA so Python can read it
        self.vidconv_out = Gst.ElementFactory.make("nvvideoconvert", "convertor_out")
        self.capsfilter_out = Gst.ElementFactory.make("capsfilter", "capsfilter_rgba")
        caps_rgba = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        self.capsfilter_out.set_property("caps", caps_rgba)

        self.sink = Gst.ElementFactory.make("fakesink", "fakesink")

        # Add all elements to pipeline
        for elem in [self.source, self.vidconv_src, self.mux, self.nvinfer, self.vidconv_out, self.capsfilter_out, self.sink]:
            self.pipeline.add(elem)

        # Link elements: source -> vidconv_src -> mux -> nvinfer -> vidconv_out -> capsfilter -> sink
        self.source.link(self.vidconv_src)
        vidconv_src_pad = self.vidconv_src.get_static_pad("src")
        mux_sink_pad = self.mux.get_request_pad("sink_0")
        vidconv_src_pad.link(mux_sink_pad)
        
        self.mux.link(self.nvinfer)
        self.nvinfer.link(self.vidconv_out)
        self.vidconv_out.link(self.capsfilter_out)
        self.capsfilter_out.link(self.sink)

        # Attach Probe to the end of the capsfilter so RGBA format is guaranteed
        probe_pad = self.capsfilter_out.get_static_pad("src")
        probe_pad.add_probe(Gst.PadProbeType.BUFFER, self.metadata_probe, 0)

        self.camera_right_yaw = -self.camera_fov / 2
        self.camera_left_yaw = self.camera_fov / 2

        # Publishers
        self.people_pub = self.create_publisher(CamPersonDetectionArray, 'cam_people_detections', 10)
        if self.debug_mode:
            self.debug_image_pub = self.create_publisher(CompressedImage, 'cam_people_detections/debug_image/compressed', 10)

        self.people_msg =  CamPersonDetectionArray()
        ros_namespace = self.get_namespace().strip('/')
        self.people_msg.header.frame_id = f'{ros_namespace}/head_link' if ros_namespace else 'head_link'

        self.pipeline.set_state(Gst.State.PLAYING)
        self.get_logger().info("DeepStream Pipeline Running!")

    def image_callback(self, msg):
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
        msg = Pose()
        msg.position.x = float(x)
        msg.position.y = float(y)
        msg.position.z = 0.0
        return msg
    
    def cam_to_angle(self, X):
        X_inv = 1 - X # Invert X to match the robot's coordinate system rather than the camera's coordinate system
        angle = (1 - X_inv) * self.camera_right_yaw + X_inv * self.camera_left_yaw # direction: right to left increase
        self.get_logger().info(f"####### Calculated angle: {angle} from X: {X} with camera FOV: {math.degrees(self.camera_fov)} degrees")
        return angle


    def metadata_probe(self, pad, info, u_data):
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

            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_copy = np.array(n_frame, copy=True, order='C')
            debug_img = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                
                mask_params = obj_meta.mask_params
                
                if mask_params.size > 0:
                    raw_data = mask_params.get_mask_array()
                    keypoints = np.array(raw_data).flatten()[:51].reshape((17, 3))
                    
                    person_msg = CamPersonDetection()
                    
                    # Store data into ROS message
                    person_msg.keypoints.nose = self.XYN_to_Pose(keypoints[0][1] / self.camera_width, (keypoints[0][2] - self.Y_padding) / self.camera_height)
                    person_msg.keypoints.left_eye = self.XYN_to_Pose(keypoints[1][1] / self.camera_width, (keypoints[1][2] - self.Y_padding) / self.camera_height)
                    person_msg.keypoints.right_eye = self.XYN_to_Pose(keypoints[2][1] / self.camera_width, (keypoints[2][2] - self.Y_padding) / self.camera_height)
                    person_msg.keypoints.left_ear = self.XYN_to_Pose(keypoints[3][1] / self.camera_width, (keypoints[3][2] - self.Y_padding) / self.camera_height)
                    person_msg.keypoints.right_ear = self.XYN_to_Pose(keypoints[4][1] / self.camera_width, (keypoints[4][2] - self.Y_padding) / self.camera_height)
                    person_msg.keypoints.left_shoulder = self.XYN_to_Pose(keypoints[5][1] / self.camera_width, (keypoints[5][2] - self.Y_padding) / self.camera_height)
                    person_msg.keypoints.right_shoulder = self.XYN_to_Pose(keypoints[6][1] / self.camera_width, (keypoints[6][2] - self.Y_padding) / self.camera_height)
                    person_msg.keypoints.left_elbow = self.XYN_to_Pose(keypoints[7][1] / self.camera_width, (keypoints[7][2] - self.Y_padding) / self.camera_height)
                    person_msg.keypoints.right_elbow = self.XYN_to_Pose(keypoints[8][1] / self.camera_width, (keypoints[8][2] - self.Y_padding) / self.camera_height)
                    person_msg.keypoints.left_wrist = self.XYN_to_Pose(keypoints[9][1] / self.camera_width, (keypoints[9][2] - self.Y_padding) / self.camera_height)
                    person_msg.keypoints.right_wrist = self.XYN_to_Pose(keypoints[10][1] / self.camera_width, (keypoints[10][2] - self.Y_padding) / self.camera_height)
                    person_msg.keypoints.left_hip = self.XYN_to_Pose(keypoints[11][1] / self.camera_width, (keypoints[11][2] - self.Y_padding) / self.camera_height)
                    person_msg.keypoints.right_hip = self.XYN_to_Pose(keypoints[12][1] / self.camera_width, (keypoints[12][2] - self.Y_padding) / self.camera_height)
                    person_msg.keypoints.left_knee = self.XYN_to_Pose(keypoints[13][1] / self.camera_width, (keypoints[13][2] - self.Y_padding) / self.camera_height)
                    person_msg.keypoints.right_knee = self.XYN_to_Pose(keypoints[14][1] / self.camera_width, (keypoints[14][2] - self.Y_padding) / self.camera_height)
                    person_msg.keypoints.left_ankle = self.XYN_to_Pose(keypoints[15][1] / self.camera_width, (keypoints[15][2] - self.Y_padding) / self.camera_height)
                    person_msg.keypoints.right_ankle = self.XYN_to_Pose(keypoints[16][1] / self.camera_width, (keypoints[16][2] - self.Y_padding) / self.camera_height)
                    
                    X_min = keypoints[:, 1].min() / self.camera_width
                    X_max = keypoints[:, 1].max() / self.camera_width

                    person_msg.bound_angle_min = Float(data=self.cam_to_angle(X_min))
                    person_msg.bound_angle_max = Float(data=self.cam_to_angle(X_max))

                    detected_people.append(person_msg)
                    if self.debug_mode:
                        # --- DRAWING THE DEBUG VISUALIZATION ---
                        # 1. Draw Bounding Box
                        x1 = int(obj_meta.rect_params.left)
                        y1 = int(obj_meta.rect_params.top)
                        w = int(obj_meta.rect_params.width)
                        h = int(obj_meta.rect_params.height)
                        cv2.rectangle(debug_img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)

                        # 2. Draw Skeleton Lines
                        for p1, p2 in SKELETON_CONNECTIONS:
                            conf_p1, x_p1, y_p1 = keypoints[p1]
                            conf_p2, x_p2, y_p2 = keypoints[p2]
                            self.get_logger().info(f"Drawing line between keypoints {x_p1};{y_p1} and {x_p2};{y_p2} with confidences {conf_p1:.2f}, {conf_p2:.2f}")
                            # Only draw the line if both keypoints are fairly confident
                            if conf_p1 > 0.2 and conf_p2 > 0.2:
                                cv2.line(debug_img, (int(x_p1), int(y_p1- self.Y_padding)), (int(x_p2), int(y_p2- self.Y_padding)), (0, 255, 255), 2)

                        # 3. Draw Keypoint Dots
                        for i in range(17):
                            kconf, kx, ky = keypoints[i]
                            if kconf > 0.2:
                                cv2.circle(debug_img, (int(kx), int(ky- self.Y_padding)), 4, (0, 255, 0), -1)

                l_obj = l_obj.next
            
            if self.debug_mode:
                debug_msg = CompressedImage()
                debug_msg.header.stamp = self.get_clock().now().to_msg()
                debug_msg.format = "jpeg"
                _, encoded_img = cv2.imencode('.jpg', debug_img)
                debug_msg.data = encoded_img.tobytes()
                self.debug_image_pub.publish(debug_msg)

            # Very important to prevent memory leaks on Jetson hardware!
            try:
                pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            except AttributeError:
                # Fails gracefully if you are on an older DeepStream version that doesn't mandate unmapping
                pass

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        # Publish the Metadata back to ROS
        if detected_people:
            self.get_logger().info(f"Detected {len(detected_people)} people in the frame.")           
            self.people_msg.header.stamp = self.get_clock().now().to_msg()
            self.people_msg.people = detected_people
            self.people_pub.publish(self.people_msg)

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