import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, ReliabilityPolicy
from mecanumbot_msgs.msg import CamPersonDetectionArray
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, Point, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from tf2_ros import TransformListener, Buffer
from transforms3d.euler import euler2quat
import math
import numpy as np
import copy

class PersonLocateNode(Node):
    def __init__(self):
        super().__init__('mecanumbot_locate_detections')

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.namespace = self.get_namespace().strip('/')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Declare parameter for the "X" meter offset behind the obstacle
        self.declare_parameter('obstacle_buffer_x', 0.5)
        
        # Publishers
        self.people_pub = self.create_publisher(PoseArray, 'people_fusion', 10)
        
        # Subscribers
        self.cam_people_sub = self.create_subscription(
            CamPersonDetectionArray, 'cam_people_detections', self.cam_people_callback, 10)
        self.laser_people_sub = self.create_subscription(
            PoseArray, 'dets', self.lidar_people_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos)
            
        # Map sub uses Transient Local QoS because maps are usually published once
        map_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, map_qos)
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.amcl_callback, 10)
            
        # State variables
        self.laser_detections = []
        self.laser_angles = []
        self.cam_detections = []
        self.scan_data = None
        self.map_data = None
        self.map_array = None
        self.amcl_pose = None

        self.people_left_FOV = PoseArray()
        self.people_left_FOV.header.frame_id = f'{self.get_namespace().strip("/")}/head_link' if self.get_namespace().strip('/') else 'head_link'
        self.people_right_FOV = PoseArray()
        self.people_right_FOV.header.frame_id = f'{self.get_namespace().strip("/")}/head_link' if self.get_namespace().strip('/') else 'head_link'
        self.people_left_FOV_pub = self.create_publisher(PoseArray, 'cam_people_detections/left_FOV', 10)
        self.people_right_FOV_pub = self.create_publisher(PoseArray, 'cam_people_detections/right_FOV', 10)
        
        self.get_logger().info("Person Locate Node has started.")

    def cam_people_callback(self, msg):
        self.cam_stamp = msg.header.stamp
        self.cam_detections = msg.people

    def scan_callback(self, msg):
        self.scan_data = msg
        
    def amcl_callback(self, msg):
        self.amcl_pose = msg.pose.pose

    def map_callback(self, msg):
        self.map_data = msg
        # Convert map 1D array to 2D numpy array for fast spatial lookups
        self.map_array = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

    def fill_bound_angle(self, X_min, X_max):
        if self.amcl_pose is None:
            self.get_logger().warn("AMCL pose is not available yet. Cannot fill FOV bounds.")
            return

        # 1. Use deepcopy so we don't accidentally modify the actual amcl_pose
        min_pose = copy.deepcopy(self.amcl_pose)
        max_pose = copy.deepcopy(self.amcl_pose)

        # 2. Extract current orientation for transforms3d: [w, x, y, z] (W is FIRST!)
        q_current = [
            self.amcl_pose.orientation.w,
            self.amcl_pose.orientation.x,
            self.amcl_pose.orientation.y,
            self.amcl_pose.orientation.z
        ]

        # 3. Convert current quaternion to Euler angles (returns roll, pitch, yaw)
        roll, pitch, yaw = transforms3d.euler.quat2euler(q_current)

        # 4. Add the relative yaw angles
        yaw_min = yaw + X_min
        yaw_max = yaw + X_max

        # 5. Convert back to quaternions: returns [w, x, y, z]
        q_min = transforms3d.euler.euler2quat(roll, pitch, yaw_min)
        q_max = transforms3d.euler.euler2quat(roll, pitch, yaw_max)

        # 6. Assign the new values back to our copied ROS poses (W is q[0])
        min_pose.orientation.w = q_min[0]
        min_pose.orientation.x = q_min[1]
        min_pose.orientation.y = q_min[2]
        min_pose.orientation.z = q_min[3]

        max_pose.orientation.w = q_max[0]
        max_pose.orientation.x = q_max[1]
        max_pose.orientation.y = q_max[2]
        max_pose.orientation.z = q_max[3]

        # 7. Append to your PoseArrays
        self.people_left_FOV.poses.append(min_pose)
        self.people_right_FOV.poses.append(max_pose)
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
        ang_min = min(person.bound_angle_min.data, person.bound_angle_max.data)
        ang_max = max(person.bound_angle_min.data, person.bound_angle_max.data)
        
        pose_candidates = []
        for laser_pose, angle in zip(self.laser_detections, self.laser_angles):
            if ang_min <= angle <= ang_max:
                pose_candidates.append(laser_pose)
                
        return pose_candidates[0] if pose_candidates else None
    
    def extrap_from_raw_scan(self, person):
        if self.scan_data is None:
            return None
            
        ranges = np.array(self.scan_data.ranges)
        ang_min_scan = self.scan_data.angle_min
        ang_inc = self.scan_data.angle_increment
        
        # Order the person bounding angles correctly
        p_min = min(person.bound_angle_min.data,person.bound_angle_max.data)
        p_max = max(person.bound_angle_min.data,person.bound_angle_max.data)

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
        self.get_logger().info(f"Extrapolated detection at local coordinates: ({x:.2f}, {y:.2f})")
        return Pose(position=Point(x=x, y=y, z=0.0))

    def handle_map_occlusion(self, local_pose):
        """ Checks if the proposed local point lands in a map obstacle and extrudes it. """
        if self.map_data is None or self.map_array is None or self.amcl_pose is None:
            return local_pose # Missing data, return standard point safely
            
        local_x = local_pose.position.x
        local_y = local_pose.position.y
        
        # 1. Get Robot global pose in map
        rx = self.amcl_pose.position.x
        ry = self.amcl_pose.position.y
        q = self.amcl_pose.orientation
        ryaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        
        # 2. Transform hit location to global map frame
        hit_x = rx + local_x * math.cos(ryaw) - local_y * math.sin(ryaw)
        hit_y = ry + local_x * math.sin(ryaw) + local_y * math.cos(ryaw)
        
        # 3. Convert to map grid coordinates
        res = self.map_data.info.resolution
        ox = self.map_data.info.origin.position.x
        oy = self.map_data.info.origin.position.y
        width = self.map_data.info.width
        height = self.map_data.info.height
        
        gx = int((hit_x - ox) / res)
        gy = int((hit_y - oy) / res)
        
        if not (0 <= gx < width and 0 <= gy < height):
            self.get_logger().warn("Detection is out of map bounds, skipping occlusion check.")
            return local_pose # Out of map bounds
            
        # 4. Check if the cell is an obstacle (> 50 confidence)
        if self.map_array[gy, gx] > 50:
            self.get_logger().info("Wall occlusion detected! Tracing back of wall...")
            
            # Global ray angle from robot
            ray_yaw = ryaw + math.atan2(local_y, local_x)
            step_size = res / 2.0 # Sub-cell stepping to ensure we don't jump gaps
            
            curr_dist = math.hypot(local_x, local_y) # Distance from robot to wall hit
            max_dist = curr_dist + 4.0 # Limit tracing to prevent infinite loops (max 4m thick wall)
            
            # Trace until free space is found
            while curr_dist < max_dist:
                curr_x_map = rx + curr_dist * math.cos(ray_yaw)
                curr_y_map = ry + curr_dist * math.sin(ray_yaw)
                
                cgx = int((curr_x_map - ox) / res)
                cgy = int((curr_y_map - oy) / res)
                
                if not (0 <= cgx < width and 0 <= cgy < height):
                    break # Ray left the map
                    
                cell_val = self.map_array[cgy, cgx]
                if cell_val < 50 and cell_val != -1: 
                    # Found free space behind wall (ignoring unknown space (-1))
                    break
                    
                curr_dist += step_size
                
            # 5. Apply the +X offset from behind the wall
            x_offset = self.get_parameter('obstacle_buffer_x').value
            final_dist = curr_dist + x_offset
            
            # 6. Re-calculate returning pose in robot's local base_link frame
            local_angle = math.atan2(local_y, local_x)
            corrected_pose = Pose()
            corrected_pose.position.x = final_dist * math.cos(local_angle)
            corrected_pose.position.y = final_dist * math.sin(local_angle)
            corrected_pose.position.z = 0.0
            
            self.get_logger().info(
                f"Corrected pose shifted +{x_offset}m behind obstacle to "
                f"({corrected_pose.position.x:.2f}, {corrected_pose.position.y:.2f})"
            )
            return corrected_pose
        self.get_logger().info("No wall occlusion detected, keeping original pose.")
        return local_pose
        
    def merge_detections(self):
        if not self.cam_detections:
            return
        self.get_logger().info(f"Fusing {len(self.cam_detections)} camera detections with {len(self.laser_detections)} LiDAR detections.")
        fused_poses = PoseArray()
        fused_poses.header.stamp = self.cam_stamp
        fused_poses.header.frame_id = 'mecanumbot/base_link'

        for person in self.cam_detections:
            # 1. Try to match with existing LiDAR detections
            person_pose = self.arrange_with_scan_dets(person)    
            
            # 2. Fallback: Extrapolate from raw scan
            if person_pose is None:
                person_pose = self.extrap_from_raw_scan(person)
                
                
            # 3. Validation: Verify pose isn't on a mapped wall
            if person_pose is not None:
                person_pose = self.handle_map_occlusion(person_pose)
                fused_poses.poses.append(person_pose)

            self.fill_bound_angle(person.bound_angle_min.data, person.bound_angle_max.data)   
        # Publish combined array
        if fused_poses.poses:
            self.get_logger().info(f"Publishing {len(fused_poses.poses)} fused detections.")
            self.people_pub.publish(fused_poses)
        if self.people_left_FOV.poses:
            self.get_logger().info(f"Publishing {len(self.people_left_FOV.poses)} left FOV detections.")
            self.people_left_FOV.stamp = self.cam_stamp 
            self.people_left_FOV_pub.publish(self.people_left_FOV)
            self.people_left_FOV.poses.clear()
        if self.people_right_FOV.poses:
            self.get_logger().info(f"Publishing {len(self.people_right_FOV.poses)} right FOV detections.")
            self.people_right_FOV.stamp = self.cam_stamp
            self.people_right_FOV_pub.publish(self.people_right_FOV)
            self.people_right_FOV.poses.clear()
            
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