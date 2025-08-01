"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-summer-labs

File Name: imu.py

Title: IMU Pose, Velocity and Attitude

Author: TEAM 1

Purpose: The goal of this lab is to build and deploy a ROS node that can ingest
IMU data and return accurate attitude estimates (roll, pitch, yaw) that can then
be used for autonomous navigation. It is recommended to review the equations of
motion and axes directions for the RACECAR Neo platform before starting. Template
code has been provided for the implementation of a Complementary Filter.

Expected Outcome: Subscribe to the /imu and /mag topics, and publish to the /attitude
topic with accurate attitude estimations.
"""

# run this straight in JupyterNB or using an external adapter specifically 
# for ROS2 - won't run directly in VSC
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField, LaserScan
from geometry_msgs.msg import Vector3
import cv2
from collections import namedtuple
import numpy as np
import math
import time

# Container to carry the state between calls (previous image, points and time)
FlowState = namedtuple("FlowState",
                       ["prev_img",   # previous LiDAR gray image (uint8)
                        "prev_pts",   # Nx1x2 float32 array of Shi-Tomasi points from prev_img
                        "prev_time"]) # timestamp (float, seconds)
# sys.path.insert(1, '../../library')
# import racecar_core

########################################################################################
# Global variables
########################################################################################
# import racecar_utils as rc_utils

# rc = racecar_core.create_racecar()

class CompFilterNode(Node):
    def __init__(self):
        super().__init__('complementary_filter_node')
        # self.get_logger().info("hello")
        # Set up subscriber and publisher nodes
        self.subscription_imu = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.subscription_mag = self.create_subscription(MagneticField, '/mag', self.mag_callback, 10)
        self.subscription_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.publisher_attitude = self.create_publisher(Vector3, '/attitude', 10) # output as [roll, pitch, yaw] angles
        self.publisher_velocity = self.create_publisher(Vector3, '/velocity', 10)
        self.publisher_pose_est = self.create_publisher(Vector3, '/pose_estimate', 10)

        self.prev_time = self.get_clock().now() # initialize time checkpoint
        self.alpha = .95 # TODO: Determine an alpha value that works with the complementary filter
        
        self.flow_state = None      # holds prev_img/pts/time for lidar_flow_velocity
        self.v_lidar = np.zeros(2)  # latest LiDAR-derived (vx, vy)

        # set up attitude params
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.mag = None

        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        self.x = 0.0
        self.y = 0.0

    # [FUNCTION] Called when new IMU data is received, attidude calc completed here as well
    def imu_callback(self, data):
        # TODO: Grab linear acceleration and angular velocity values from subscribed data points
        accel = data.linear_acceleration
        gyro = data.angular_velocity
        ax, ay, az = data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z
        accel_array = [ax, ay, az] # make the raw accelerations into an array
        gx, gy, gz = data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z
        # TODO: Calculate time delta
        now = self.get_clock().now() # Current ROS time
        dt = (now - self.prev_time).nanoseconds * 1e-9 # Time delta
        self.prev_time = now # refresh checkpoint

        # Attitude angle derivations, see full formula here:
        # https://ahrs.readthedocs.io/en/latest/filters/complementary.html
    
        # TODO: Derive tilt angles from accelerometer
        accel_roll = math.atan2(accel.y, accel.x) # theta_x
        accel_pitch = math.atan2(-1*accel.x, (math.sqrt(accel.y**2 + accel.z**2))) # theta_y - seems correct
        # accel_yaw

        # TODO: Integrate gyroscope to get attitude angles
        gyro_roll = self.roll + gx * dt # theta_xt
        gyro_pitch = self.pitch + gy * dt # theta_yt
        gyro_yaw = self.yaw + gz * dt # theta_zt

        # TODO: Compute yaw angle from magnetometer
        if self.mag:
            mx, my, mz = self.mag
            print(f"Mag norm (~50 uT): {math.sqrt(mx**2 + my**2 + mz**2) * 1e6}") # used for checking magnetic disturbances/offsets
            bx = mx*math.cos(accel_pitch) + mz*math.sin(accel_pitch)
            by = mx*math.sin(accel_roll) * math.sin(accel_pitch) + my*math.cos(accel_roll) - mz*math.sin(accel_roll)*math.cos(accel_roll)
            
            mag_accel_yaw = math.atan2(by, bx) + math.pi
        else:
            mag_accel_yaw = self.yaw
        
        # TODO: Fuse gyro, mag, and accel derivations in complemtnary filter
        self.roll = self.alpha*gyro_roll + (1-self.alpha)*accel_pitch
        self.pitch = self.alpha*gyro_pitch + (1-self.alpha)*accel_roll
        self.yaw = self.alpha*gyro_yaw + (1-.995)*mag_accel_yaw

        g = -9.81
        gravity_array = np.array([
            g * np.sin(self.pitch), 
            -g * np.sin(self.roll), 
            g * np.cos(self.pitch) * np.cos(self.roll)
        ]) # gravity components in each axis into an array

        no_grav_accel = accel_array - gravity_array
        #L.A. integration
        la_velocity = np.array([
            self.vx + ax * dt,
            self.vy + ay * dt,
            self.vz + az * dt
        ])
        la_vx, la_vy, la_vz = la_velocity[0], la_velocity[1], la_velocity[2] 

        # if self.lidar:
        #     scan = self.lidar
        # else:
        #     scan = 0
        
        # ---------------------------------------------
        # Fuse IMU-integrated velocity with LiDAR flow
        # ---------------------------------------------
        imu_velocity = np.array([la_vx, la_vy])      # estimate from L.A.
        alpha_v = 0.95 #trust L.A. integration more than the freaking Lidar                          
        v_fused = alpha_v * imu_velocity + (1 - alpha_v) * self.v_lidar
        self.vx, self.vy = v_fused               # keep running copy

        c, s = math.cos(self.yaw), math.sin(self.yaw)
        vx_w =  c * v_fused[0] - s * v_fused[1]   # east-ish   (m s-1)
        vy_w =  s * v_fused[0] + c * v_fused[1]   # north-ish  (m s-1)

        # --------  dead-reckoning integration  --------
        self.x += vx_w * dt
        self.y += vy_w * dt
        
        # Print results for sanity checking
        print(f"====== Complementary Filter Results ======")
        print(f"Speed || Freq = {round(1/dt,0)} || dt (ms) = {round(dt*1e3, 2)}")
        print(f"Accel + Mag Derivation")
        print(f"Roll (deg): {accel_roll * 180/math.pi}")
        print(f"Pitch (deg): {accel_pitch * 180/math.pi}")
        print(f"Yaw (deg): {mag_accel_yaw * 180/math.pi}")
        print()
        print(f"Gyro Derivation")
        print(f"Roll (deg): {gyro_roll * 180/math.pi}")
        print(f"Pitch (deg): {gyro_pitch * 180/math.pi}")
        print(f"Yaw (deg): {gyro_yaw * 180/math.pi}")
        print()
        print(f"Fused Results")
        print(f"Roll (deg): {self.roll * 180/math.pi}")
        print(f"Pitch (deg): {self.pitch * 180/math.pi}")
        print(f"Yaw (deg): {self.yaw * 180/math.pi}")
        print(f"Acceleration array: {accel_array}")
        print(f"Acceleration without gravity: {no_grav_accel}")
        print(f"Vx: {v_fused[0]:+.2f} m/s      Vy: {v_fused[1]:+.2f} m/s")
        print(f"X: {self.x:+.2f}   Y: {self.y:+.2f}")
        print("\n")

        
        # TODO: Publish to attitude topic (convert to degrees)
        attitude = Vector3()
        attitude.x = self.roll * 180.0 / math.pi
        attitude.y = self.pitch * 180.0 / math.pi
        attitude.z = self.yaw * 180.0 / math.pi
        self.publisher_attitude.publish(attitude)

        # TODO:Publish to velocity topic 
        vel_msg = Vector3()
        vel_msg.x = v_fused[0]   # +x = right (body frame)
        vel_msg.y = v_fused[1]   # +y = forward
        vel_msg.z = la_vz        # keep your existing vertical estimate
        self.publisher_velocity.publish(vel_msg)

        pose_msg = Vector3()      # reuse Vector3 for convenience
        pose_msg.x = self.x       # metres east/right
        pose_msg.y = self.y       # metres north/forward
        pose_msg.z = self.yaw     # yaw in radians (leave as-is or convert)
        self.publisher_pose_est.publish(pose_msg)
    
    # [FUNCTION] Called when magnetometer topic receives an update
    def mag_callback(self, data):
        # TODO: Assign self.mag to the magnetometer data points
        self.mag = [data.magnetic_field.x, data.magnetic_field.y, data.magnetic_field.z]
        mx = self.mag[0]
        my = self.mag[1]
        mz = self.mag[2]
    
    # [FUNCTION] Called whenever lidar topic receives an update
    def lidar_callback(self, scan_msg):
        """
        Convert LaserScan ranges to cm, call lidar_flow_velocity(),
        store the result in self.v_lidar for use inside imu_callback().
        """

        # self.lidar = data.ranges
        # self.lidar = data.scan()
        # 1. ranges[] are in metres → convert to cm for helper
        lidar_scan_cm = np.array(scan_msg.ranges, dtype=np.float32) * 100.0

        # 2. Run optical flow; preserve state between frames
        v_body, self.flow_state = lidar_flow_velocity(
            lidar_scan_cm,
            state=self.flow_state,
            debug=False)   # switch to True to visualise arrows

        # 3. Save for later fusion with IMU velocity
        self.v_lidar = v_body

def lidar_flow_velocity(lidar_scan,
                        state=None,
                        max_range_cm=1000,
                        radius_px=128,
                        min_pts=30,
                        debug=False):
    """
    Estimate planar velocity from consecutive LiDAR scans using Shi-Tomasi + Lucas-Kanade.

    Parameters
    ----------
    lidar_scan : 1-D array of range samples (cm) straight from rc.lidar.get_samples()
    state      : FlowState or None.  Pass the returned value from the *previous* call.
    max_range_cm : clip distance for display + pixel-to-metre scale
    radius_px  : radius of the LiDAR image (half of img width/height in pixels)
    min_pts    : if tracked features fall below this, re-run Shi-Tomasi
    debug      : if True, shows OpenCV windows with features and flow

    Returns
    -------
    v_body_mps : (vx, vy) numpy array in m / s, **body frame**  (+x = right, +y = forward)
    new_state  : FlowState to feed into the next call
    """

    # 1. Make a gray image from the polar scan (re-using your helper)
    img = lidar_to_gray(lidar_scan, max_range_cm=max_range_cm, radius_px=radius_px)

    # 2. Take the current timestamp for Δt computation
    t_now = time.time()

    # 3. If this is the first frame → initialise state and return zero velocity
    if state is None or state.prev_img is None:
        # Run Shi-Tomasi once so next frame has points to track
        pts = cv2.goodFeaturesToTrack(img,
                                      maxCorners=300,
                                      qualityLevel=0.01,
                                      minDistance=3)
        new_state = FlowState(prev_img=img, prev_pts=pts, prev_time=t_now)
        return np.array([0.0, 0.0]), new_state

    # 4. Copy previous data out of the namedtuple
    prev_img, prev_pts, prev_time = state

    # 5. Re-detect points if too few survived
    if prev_pts is None or len(prev_pts) < min_pts:
        prev_pts = cv2.goodFeaturesToTrack(prev_img,
                                           maxCorners=300,
                                           qualityLevel=0.01,
                                           minDistance=3)

    # 6. Track those points with pyramidal Lucas–Kanade optical flow
    p1, st, _err = cv2.calcOpticalFlowPyrLK(prev_img,  # previous image
                                            img,       # current image
                                            prev_pts.astype(np.float32),
                                            None)      # let OpenCV allocate output

    # 7. Keep only the points successfully tracked (st == 1)
    good = st[:, 0] == 1
    p0g, p1g = prev_pts[good], p1[good]           # before/after coordinates

    # 8. Compute displacement in pixels for each surviving feature
    disp_pix = (p1g - p0g).reshape(-1, 2)         # shape (N,2)

    # 9. Average the pixel displacements to get a single 2-D flow vector
    mean_disp_pix = disp_pix.mean(axis=0) if len(disp_pix) else np.array([0.0, 0.0])

    # 10. Convert pixel shift → distance (metres).  Derive scale once: cm/px then to metres
    dx_cm_per_px = max_range_cm / float(radius_px)   # how many cm a 1-pixel shift represents
    trans_cm = mean_disp_pix * dx_cm_per_px          # → cm
    trans_m  = trans_cm / 100.0                      # → metres

    # 11. Compute Δt between scans and avoid divide-by-zero
    dt = max(t_now - prev_time, 1e-6)

    # 12. Velocity in body frame (+x right, +y forward)
    v_body_mps = trans_m / dt                        # metres per second

    # 13. (Optional) debug visualisation
    if debug:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for (x0, y0), (x1, y1) in zip(p0g.reshape(-1, 2), p1g.reshape(-1, 2)):
            cv2.arrowedLine(vis, (int(x0), int(y0)),
                            (int(x1), int(y1)),
                            (0, 255, 0), 1, tipLength=0.2)
        cv2.imshow("LiDAR flow", vis)
        cv2.waitKey(1)

    # 14. Package new_state for next call
    new_state = FlowState(prev_img=img, prev_pts=p1g.reshape(-1, 1, 2), prev_time=t_now)

    # 15. Return velocity estimate and updated state
    return v_body_mps, new_state

def lidar_to_gray(samples_cm, max_range_cm=1000, radius_px=128):
    N = len(samples_cm)
    angles = np.linspace(-np.pi, np.pi, N, endpoint=False)
    # convert to meters if you prefer
    r = np.clip(samples_cm, 0, max_range_cm)

    x = r * np.cos(angles)
    y = r * np.sin(angles)

    img = np.zeros((2*radius_px, 2*radius_px), dtype=np.uint8)
    u = ((x / max_range_cm) * radius_px + radius_px).astype(int)
    v = ((y / max_range_cm) * radius_px + radius_px).astype(int)

    valid = (u>=0)&(u<2*radius_px)&(v>=0)&(v<2*radius_px)
    img[v[valid], u[valid]] = 255
    return img

def main():
    rclpy.init(args=None)
    node = CompFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
