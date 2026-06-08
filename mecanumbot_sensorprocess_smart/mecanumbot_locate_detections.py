import rclpy
from rclpy.node import Node
from mecanumbot_msgs.msg import CamPersonDetectionArray
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, Point
from tf2_ros import TransformListener, Buffer
import math
import numpy as np

class PersonLocateNode(Node):
    def __init__(self):
        super().__init__('mecanumbot_locate_detections')
        self.namespace = self.get_namespace().strip('/')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publisher
        self.people_pub = self.create_publisher(PoseArray, 'people_fusion', 10)
        
        # Subscribers
        self.cam_people_sub = self.create_subscription(
            CamPersonDetectionArray, 'cam_detected_people', self.cam_people_callback, 10)
        self.laser_people_sub = self.create_subscription(
            PoseArray, 'lidar_detected_people', self.lidar_people_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)
            
        self.laser_detections = []
        self.laser_angles = []
        self.cam_detections = []
        self.scan_data = None
        
        self.get_logger().info("Person Locate Node has started.")

    def cam_people_callback(self, msg):
        self.cam_detections = msg.people

    def scan_callback(self, msg):
        self.scan_data = msg

    def lidar_people_callback(self, msg):
        try:
            # Safely fetch transform, defaulting to base_scan if header is missing
            source_frame = msg.header.frame_id if msg.header.frame_id else 'mecanumbot/base_scan'
            transform = self.tf_buffer.lookup_transform(
                'mecanumbot/base_link',
                source_frame,
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=2.0)
            return

        # Extract translation and yaw
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        q = transform.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        self.laser_detections = []
        self.laser_angles = []
        
        # Fast native math transformation (avoids tf2_geometry_msgs overhead)
        for pose in msg.poses:
            global_x = tx + (pose.position.x * cos_yaw) - (pose.position.y * sin_yaw)
            global_y = ty + (pose.position.x * sin_yaw) + (pose.position.y * cos_yaw)
            
            new_pose = Pose()
            new_pose.position.x = global_x
            new_pose.position.y = global_y
            new_pose.position.z = 0.0
            
            self.laser_detections.append(new_pose)
            self.laser_angles.append(math.atan2(global_y, global_x))
            
        self.merge_detections()

    def arrange_with_scan_dets(self, person):
        # Ensure correct min/max bounds even if wrapped
        ang_min = min(person.bound_angle_min, person.bound_angle_max)
        ang_max = max(person.bound_angle_min, person.bound_angle_max)
        
        pose_candidates = []
        for laser_pose, angle in zip(self.laser_detections, self.laser_angles):
            if ang_min <= angle <= ang_max:
                pose_candidates.append(laser_pose)
                self.get_logger().debug(f"Cam det matches laser det at ({laser_pose.position.x:.2f}, {laser_pose.position.y:.2f})")
                
        # Return the first match (or implement a distance-based best fit here)
        return pose_candidates[0] if pose_candidates else None
    
    def extrap_from_raw_scan(self, person):
        if self.scan_data is None:
            return None
            
        ranges = np.array(self.scan_data.ranges)
        ang_min_scan = self.scan_data.angle_min
        ang_inc = self.scan_data.angle_increment
        
        # Order the person bounding angles correctly
        p_min = min(person.bound_angle_min, person.bound_angle_max)
        p_max = max(person.bound_angle_min, person.bound_angle_max)

        # Calculate indices and clamp them to array bounds to prevent IndexError
        idx_min = int((p_min - ang_min_scan) / ang_inc)
        idx_max = int((p_max - ang_min_scan) / ang_inc)
        
        idx_min = max(0, min(idx_min, len(ranges) - 1))
        idx_max = max(0, min(idx_max, len(ranges) - 1))

        if idx_min >= idx_max:
            return None

        # Extract distances and filter out inf, nan, and out-of-range limits
        slice_ranges = ranges[idx_min:idx_max+1]
        valid_mask = (
            (slice_ranges > self.scan_data.range_min) & 
            (slice_ranges < self.scan_data.range_max) & 
            ~np.isinf(slice_ranges) & 
            ~np.isnan(slice_ranges)
        )
        valid_ranges = slice_ranges[valid_mask]

        if len(valid_ranges) == 0:
            return None

        # Use median to ignore background laser hits
        dist_median = float(np.median(valid_ranges))
        center_angle = p_min + (p_max - p_min) / 2.0
        
        x = dist_median * math.cos(center_angle)
        y = dist_median * math.sin(center_angle)
        
        self.get_logger().debug(f"Extrapolated position: ({x:.2f}, {y:.2f}) at angle {center_angle:.2f}")
        return Pose(position=Point(x=x, y=y, z=0.0))
        
    def merge_detections(self):
        if not self.cam_detections:
            return
        
        self.get_logger().info(f"Camera detections: {len(self.cam_detections)}")
        
        fused_poses = PoseArray()
        fused_poses.header.stamp = self.get_clock().now().to_msg()
        fused_poses.header.frame_id = 'mecanumbot/base_link'

        for person in self.cam_detections:
            # 1. Try to match with existing LiDAR detections
            person_pose = self.arrange_with_scan_dets(person)    
            
            # 2. Fallback: Extrapolate from raw scan
            if person_pose is None:
                person_pose = self.extrap_from_raw_scan(person)
                
            if person_pose is not None:
                fused_poses.poses.append(person_pose)
                self.get_logger().info(f"Fused pose: ({person_pose.position.x:.2f}, {person_pose.position.y:.2f})")
            else:
                self.get_logger().info("Could not extrapolate pose for person.")
                
        # Publish combined array
        if fused_poses.poses:
            self.people_pub.publish(fused_poses)

def main(args=None):
    rclpy.init(args=args)
    node = PersonLocateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()