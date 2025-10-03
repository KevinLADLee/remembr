#!/usr/bin/env python3
"""
Convert ROS2 bag file to dataset format.
- Extract /camera/color/image_raw topic to imgs/ folder (timestamp.jpg)
- Extract /amcl_pose topic to poses.csv (timestamp, x, y, z, roll, pitch, yaw)
"""

import os
import argparse
import csv
from pathlib import Path

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
import cv2
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_to_euler(x, y, z, w):
    """Convert quaternion to roll, pitch, yaw (in radians)."""
    rotation = Rotation.from_quat([x, y, z, w])
    roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)
    return roll, pitch, yaw


def convert_rosbag_to_dataset(bag_path, output_dir, image_topic, pose_topic, image_interval=0.0, pose_interval=0.0):
    """
    Convert ROS2 bag to dataset format.

    Args:
        bag_path: Path to ROS2 bag directory
        output_dir: Output directory for dataset
        image_topic: Topic name for images
        pose_topic: Topic name for poses
        image_interval: Minimum time interval between images in seconds (0 = save all)
        pose_interval: Minimum time interval between poses in seconds (0 = save all)
    """
    # Create output directories
    output_path = Path(output_dir)
    imgs_dir = output_path / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    poses_file = output_path / "poses.csv"

    # Initialize CV Bridge for image conversion
    bridge = CvBridge()

    # Open the bag
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Get topic types
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    # Prepare poses CSV
    poses_data = []

    print(f"Processing bag: {bag_path}")
    print(f"Output directory: {output_dir}")
    print(f"Image interval: {image_interval}s, Pose interval: {pose_interval}s")

    image_count = 0
    pose_count = 0
    last_image_time = -float('inf')
    last_pose_time = -float('inf')

    # Read messages
    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        # Process image messages
        if topic == image_topic:
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            # Handle both Image and CompressedImage types
            if 'CompressedImage' in type_map[topic]:
                # Decode compressed image
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                # Convert ROS image to OpenCV image
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Use message timestamp
            sec = msg.header.stamp.sec
            nanosec = msg.header.stamp.nanosec
            timestamp_sec = sec + nanosec / 1e9

            # Check if enough time has passed since last saved image
            if timestamp_sec - last_image_time >= image_interval:
                # Save image with timestamp as filename (sec.nanosec format)
                timestamp_str = f"{sec}.{nanosec:09d}"
                img_filename = imgs_dir / f"{timestamp_str}.jpg"
                cv2.imwrite(str(img_filename), cv_image)
                image_count += 1
                last_image_time = timestamp_sec

                if image_count % 10 == 0:
                    print(f"Processed {image_count} images...")

        # Process pose messages
        elif topic == pose_topic:
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            pose = None

            # Extract pose information based on message type
            if type_map[topic] == 'geometry_msgs/msg/PoseWithCovarianceStamped':
                # PoseWithCovarianceStamped has pose.pose
                pose = msg.pose.pose
            elif type_map[topic] == 'geometry_msgs/msg/PoseStamped':
                # PoseStamped has pose directly
                pose = msg.pose
            elif type_map[topic] == 'nav_msgs/msg/Odometry':
                # Odometry has pose.pose
                pose = msg.pose.pose
            else:
                print(f"Unsupported pose message type: {type_map[topic]}")
                continue


            sec = msg.header.stamp.sec
            nanosec = msg.header.stamp.nanosec
            timestamp_sec = sec + nanosec / 1e9

            # Check if enough time has passed since last saved pose
            if timestamp_sec - last_pose_time >= pose_interval:
                x = pose.position.x
                y = pose.position.y
                z = pose.position.z

                # Convert quaternion to euler angles
                qx = pose.orientation.x
                qy = pose.orientation.y
                qz = pose.orientation.z
                qw = pose.orientation.w

                roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)

                # Store timestamp in sec.nanosec format
                timestamp_str = f"{sec}.{nanosec:09d}"
                poses_data.append([timestamp_str, x, y, z, roll, pitch, yaw])
                pose_count += 1
                last_pose_time = timestamp_sec

                if pose_count % 10 == 0:
                    print(f"Processed {pose_count} poses...")

    # Write poses to CSV
    with open(poses_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
        writer.writerows(poses_data)

    print(f"\nConversion complete!")
    print(f"Images saved: {image_count}")
    print(f"Poses saved: {pose_count}")
    print(f"Dataset location: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert ROS2 bag to dataset format')
    parser.add_argument('--bag_path', type=str, help='Path to ROS2 bag directory')
    parser.add_argument('--output_dir', type=str, help='Output directory for dataset')
    parser.add_argument('--image-topic', type=str, default='/camera/color/image_raw',
                        help='Image topic name (default: /camera/color/image_raw)')
    parser.add_argument('--pose-topic', type=str, default='/amcl_pose',
                        help='Pose topic name (default: /amcl_pose)')
    parser.add_argument('--image-interval', type=float, default=0.0,
                        help='Minimum time interval between images in seconds (default: 0.0 = save all)')
    parser.add_argument('--pose-interval', type=float, default=0.0,
                        help='Minimum time interval between poses in seconds (default: 0.0 = save all)')

    args = parser.parse_args()

    if not os.path.exists(args.bag_path):
        print(f"Error: Bag path does not exist: {args.bag_path}")
        return

    convert_rosbag_to_dataset(args.bag_path, args.output_dir, args.image_topic, args.pose_topic,
                              args.image_interval, args.pose_interval)


if __name__ == '__main__':
    main()
