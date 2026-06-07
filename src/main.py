"""
Autonomous Drone - Obstacle Detection & Avoidance
BVRIT Hyderabad College of Engineering for Women
Team: Kalluri Hansika, Dhatri Sri Singu, K. Harika, S. Sanika
"""

import time
import pyrealsense2 as rs
import numpy as np
import cv2
from dronekit import connect, VehicleMode
from pymavlink import mavutil
import os
from datetime import datetime


# ============================================================
# DEVICE INFO
# ============================================================
def print_device_info(device):
    """Prints RealSense camera details."""
    print("RealSense Device Information:")
    print(f"  Name: {device.get_info(rs.camera_info.name)}")
    print(f"  Serial Number: {device.get_info(rs.camera_info.serial_number)}")
    print(f"  Firmware Version: {device.get_info(rs.camera_info.firmware_version)}")
    print(f"  USB Type: {device.get_info(rs.camera_info.usb_type_descriptor)}")


# ============================================================
# REALSENSE SETUP
# ============================================================
def start_realsense_pipeline():
    """Initializes and starts the RealSense depth + color camera stream."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"Failed to start RealSense pipeline: {e}")
        return None, None, None

    device = profile.get_device()
    print_device_info(device)

    try:
        depth_sensor = device.first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
    except Exception as e:
        print(f"Error getting depth sensor: {e}")
        pipeline.stop()
        return None, None, None

    print(f"Depth scale: {depth_scale}")
    return pipeline, depth_scale, profile


# ============================================================
# SAVE DETECTION IMAGES
# ============================================================
def save_detection_images(depth_frame, color_frame):
    """Saves color image, depth colormap, and point cloud on obstacle detection."""
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("detections", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "detected_depth.npy"), depth_image)
    cv2.imwrite(os.path.join(save_dir, "detected_color.png"), color_image)
    cv2.imwrite(os.path.join(save_dir, "detected_depth_colormap.png"), depth_colormap)

    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    points.export_to_ply(os.path.join(save_dir, "detected_pointcloud.ply"), color_frame)

    print(f"Saved detection data to: {save_dir}")


# ============================================================
# DRONE MOVEMENT COMMANDS
# ============================================================
def send_body_velocity(vehicle, vx, vy, vz):
    """Sends velocity command to drone in body frame."""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def stop_motion(vehicle):
    send_body_velocity(vehicle, 0, 0, 0)


def roll_right(vehicle, rate_rad_s=0.3):
    thrust = 0.5
    msg = vehicle.message_factory.set_attitude_target_encode(
        0, 0, 0,
        0b00000100,
        [0, 0, 0, 0],
        rate_rad_s, 0, 0,
        thrust
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def pitch_forward(vehicle, rate_rad_s=0.3):
    thrust = 0.5
    msg = vehicle.message_factory.set_attitude_target_encode(
        0, 0, 0,
        0b00000100,
        [0, 0, 0, 0],
        0, -rate_rad_s, 0,
        thrust
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


# ============================================================
# TAKEOFF
# ============================================================
def wait_for_manual_arm_and_takeoff(vehicle, target_alt=2.0):
    """Waits for manual arm, then commands takeoff to target altitude."""
    print("Waiting for vehicle to be armed and in ALT_HOLD mode...")
    while not vehicle.armed or vehicle.mode.name != "ALT_HOLD":
        print(f"  Armed: {vehicle.armed}, Mode: {vehicle.mode.name}")
        time.sleep(1)

    print("Vehicle armed. Taking off...")
    vehicle.simple_takeoff(target_alt)

    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"  Altitude: {alt:.2f} m")
        if alt >= target_alt * 0.95:
            print("Target altitude reached.")
            break
        time.sleep(1)


# ============================================================
# OBSTACLE DETECTION
# ============================================================
def detect_large_object(depth_frame, depth_scale, max_distance=0.8, min_area_pixels=5000):
    """
    Detects a large nearby object using depth image.
    Returns: (detected, cx, cy, area_m2, distance)
    """
    depth_image = np.asanyarray(depth_frame.get_data())
    max_dist_units = max_distance / depth_scale

    depth_filtered = np.where(depth_image > 0, depth_image, 999999)
    mask = np.array(depth_filtered < max_dist_units, dtype=np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area_pixels = cv2.contourArea(largest)

        if area_pixels >= min_area_pixels:
            M = cv2.moments(largest)
            if M["m00"] == 0:
                return False, None, None, None, None

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            distance = depth_frame.get_distance(cx, cy)

            intr = depth_frame.profile.as_video_stream_profile().intrinsics
            fx, fy = intr.fx, intr.fy
            x, y, w, h = cv2.boundingRect(largest)
            width_m = (w * distance) / fx
            height_m = (h * distance) / fy
            approx_area_m2 = width_m * height_m

            return True, cx, cy, approx_area_m2, distance

    return False, None, None, None, None


# ============================================================
# MAIN
# ============================================================
def main():
    # Initialize RealSense
    pipeline, depth_scale, profile = start_realsense_pipeline()
    if pipeline is None:
        print("Exiting: RealSense initialization failed.")
        return

    print("Warming up camera...")
    time.sleep(3)

    # Connect to drone via serial
    print("Connecting to drone...")
    vehicle = connect('/dev/ttyACM0', baud=57600, wait_ready=True)
    print("Connected.")

    try:
        wait_for_manual_arm_and_takeoff(vehicle, target_alt=1.3)

        print("Moving forward at 0.3 m/s...")
        send_body_velocity(vehicle, 0.3, 0, 0)

        sheet_detected = False
        roll_done = False
        timeout_retry_count = 0
        MAX_TIMEOUT_RETRIES = 10

        while True:
            # Capture frames
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
                timeout_retry_count = 0
            except RuntimeError:
                timeout_retry_count += 1
                print("Frame timeout. Retrying...")
                if timeout_retry_count >= MAX_TIMEOUT_RETRIES:
                    print("Too many timeouts. Restarting pipeline.")
                    pipeline.stop()
                    time.sleep(1)
                    pipeline, depth_scale, profile = start_realsense_pipeline()
                continue

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                print("Missing frame, skipping.")
                continue

            # Obstacle detection
            detected, cx, cy, area_m2, dist = detect_large_object(depth_frame, depth_scale)
            if detected:
                print(f"Object detected at {dist:.2f} m, area ~{area_m2:.3f} m²")
            else:
                print("No object detected.")

            # Stop and save if obstacle is close
            if detected and dist <= 0.5 and area_m2 >= 0.05 and not sheet_detected:
                print("Obstacle too close! Stopping drone.")
                stop_motion(vehicle)
                save_detection_images(depth_frame, color_frame)
                sheet_detected = True
                time.sleep(1)

            # Avoidance: roll right until obstacle cleared
            if sheet_detected and not roll_done:
                print("Avoiding obstacle by rolling right...")
                while True:
                    try:
                        frames = pipeline.wait_for_frames(timeout_ms=2000)
                    except RuntimeError:
                        continue
                    depth_frame = frames.get_depth_frame()
                    if not depth_frame:
                        continue
                    detected, _, _, _, dist = detect_large_object(depth_frame, depth_scale)
                    if not detected or dist > 1.5:
                        print("Obstacle cleared.")
                        stop_motion(vehicle)
                        roll_done = True
                        break
                    roll_right(vehicle)
                    time.sleep(0.1)

            # Resume forward flight after avoidance
            if roll_done:
                print("Resuming forward flight...")
                for _ in range(30):
                    pitch_forward(vehicle)
                    time.sleep(0.1)
                stop_motion(vehicle)
                print("Switching to LOITER mode...")
                vehicle.mode = VehicleMode("LOITER")
                print("Drone is now hovering (LOITER).")
                while True:
                    time.sleep(1)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        print("Landing drone...")
        vehicle.mode = VehicleMode("LAND")
        time.sleep(10)
        vehicle.close()
        if pipeline:
            pipeline.stop()
        print("Done.")


if __name__ == "__main__":
    main()
