from remembr.captioners.remote_captioner import RemoteAPICaptioner

from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory
from PIL import Image as im
import concurrent
import csv
from pathlib import Path
import time
import bisect
import numpy as np


class RawDbMemoryBuilder:
    def __init__(self, dataset_path, db_ip, collection_name, api_base="http://localhost:11434/v1", caption_llm_type="qwen2.5vl:7b", enable_local_captioner=False):
        self.dataset_path = dataset_path
        self.db_ip = db_ip
        self.collection_name = collection_name
        self.caption_llm_type = caption_llm_type

        self.caption_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.memory = MilvusMemory(collection_name, db_ip=db_ip)

        if enable_local_captioner:
            from remembr.captioners.qwen_captioner import QwenVLCaptioner
            self.captioner = QwenVLCaptioner()
        else:
            self.captioner = RemoteAPICaptioner(api_base=api_base, model_type=caption_llm_type)
        
        print("Initialized RawDbMemoryBuilder")


    def run(self, time_window=2.0, pose_interval=1.0, min_displacement=0.5, min_angle_diff=10.0):
        """
        Process raw dataset and build memory.
        For each pose, find images within a time window and create a memory item.

        Args:
            time_window: Time window in seconds (before and after pose) to search for images
            pose_interval: Minimum time interval between processed poses (to avoid redundancy)
            min_displacement: Minimum spatial displacement in meters between consecutive poses
            min_angle_diff: Minimum angular difference in degrees between consecutive poses
        """
        dataset_path = Path(self.dataset_path)
        imgs_dir = dataset_path / "imgs"
        poses_file = dataset_path / "poses.csv"

        if not imgs_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {imgs_dir}")
        if not poses_file.exists():
            raise FileNotFoundError(f"Poses file not found: {poses_file}")

        # Load all poses with timestamps (sorted)
        poses_list = []
        with open(poses_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                poses_list.append({
                    'timestamp': float(row['timestamp']),
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'z': float(row['z']),
                    'yaw': float(row['yaw'])
                })

        # Load all images with timestamps and create a sorted list
        image_files = sorted(list(imgs_dir.glob("*.jpg")))
        images_list = []
        image_timestamps = []
        for img_file in image_files:
            timestamp = float(img_file.stem)
            images_list.append({
                'timestamp': timestamp,
                'path': img_file
            })
            image_timestamps.append(timestamp)

        print(f"Found {len(images_list)} images and {len(poses_list)} poses")
        print(f"Processing with time_window={time_window}s, pose_interval={pose_interval}s")
        print(f"Spatial filtering: min_displacement={min_displacement}m, min_angle_diff={min_angle_diff}°")

        if len(poses_list) == 0 or len(images_list) == 0:
            print("No poses or images to process")
            return

        # Sample poses with temporal and spatial filtering
        sampled_poses = []
        last_processed_time = -float('inf')
        last_pose = None

        for pose in poses_list:
            # Temporal filtering
            if pose['timestamp'] - last_processed_time < pose_interval:
                continue

            # Spatial filtering
            if last_pose is not None:
                # Calculate displacement
                dx = pose['x'] - last_pose['x']
                dy = pose['y'] - last_pose['y']
                dz = pose['z'] - last_pose['z']
                displacement = np.sqrt(dx**2 + dy**2 + dz**2)

                # Calculate angular difference (in degrees)
                angle_diff = abs(np.degrees(pose['yaw'] - last_pose['yaw']))
                # Normalize to [0, 180]
                angle_diff = min(angle_diff, 360 - angle_diff)

                # Skip if both displacement and angle are too small
                if displacement < min_displacement and angle_diff < min_angle_diff:
                    continue

            sampled_poses.append(pose)
            last_processed_time = pose['timestamp']
            last_pose = pose

        print(f"Sampled {len(sampled_poses)} poses from {len(poses_list)} total poses")

        # Build segments: for each pose, find images within time window
        segments = []
        for pose in sampled_poses:
            pose_time = pose['timestamp']

            # Find images within [pose_time - time_window, pose_time + time_window]
            start_idx = bisect.bisect_left(image_timestamps, pose_time - time_window)
            end_idx = bisect.bisect_right(image_timestamps, pose_time + time_window)

            segment_images = images_list[start_idx:end_idx]

            if len(segment_images) > 0:
                segments.append((segment_images, [pose]))

        print(f"Created {len(segments)} segments, processing sequentially...")

        # Process segments sequentially to avoid Ollama 500 errors
        for idx, (seg_images, seg_poses) in enumerate(segments):
            self._process_segment(seg_images, seg_poses, idx)

        print(f"Finished processing {len(segments)} segments into memory database")

    def _process_segment(self, images_data, poses_data, segment_idx):
        """Process a single time segment of images and poses."""
        if len(images_data) == 0:
            return

        # Limit to max 3 images per segment, evenly sampled
        if len(images_data) > 3:
            # Sample 3 images evenly distributed
            indices = np.linspace(0, len(images_data) - 1, 3, dtype=int)
            sampled_images = [images_data[i] for i in indices]
        else:
            sampled_images = images_data

        # Load sampled images
        images = []
        for img_data in sampled_images:
            img = im.open(img_data['path'])
            images.append(img)

        # Calculate average pose from all poses in this segment
        if len(poses_data) > 0:
            avg_x = sum(p['x'] for p in poses_data) / len(poses_data)
            avg_y = sum(p['y'] for p in poses_data) / len(poses_data)
            # avg_z = sum(p['z'] for p in poses_data) / len(poses_data)
            avg_yaw = sum(p['yaw'] for p in poses_data) / len(poses_data)
        else:
            # No pose data for this segment, skip
            print(f"Warning: Segment {segment_idx} has no pose data, skipping")
            return

        # Use middle timestamp
        mid_time = (sampled_images[0]['timestamp'] + sampled_images[-1]['timestamp']) / 2

        # Caption images
        print(f"Processing segment {segment_idx + 1}: {len(images)} images (from {len(images_data)} total), {len(poses_data)} poses...")
        start_time = time.time()
        # Retry captioning up to 3 times on failure
        max_retries = 3
        caption_text = None
        for attempt in range(max_retries):
            try:
                caption_text = self.captioner.caption(images)
                break  # Success, exit retry loop
            except Exception as e:
                print(f"Captioning attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print(f"All {max_retries} captioning attempts failed, skipping segment {segment_idx}")
                return
            time.sleep(1)  # Wait 1 second before retry
        caption_time = time.time() - start_time
        print(f"Caption: {caption_text[:100]}... (took {caption_time:.2f}s)")

        # Create and insert memory item
        entity = MemoryItem(
            position=[avg_x, avg_y, avg_yaw],
            theta=avg_yaw,
            time=mid_time,
            caption=caption_text
        )

        # Debugging info
        debug = True
        if(debug):
            # Log position and caption and time in ddyymm_hhmmss format into file
            log_file = Path("/tmp/memory_insert_log.txt")
            time_struct = time.localtime(mid_time)
            time_str = time.strftime("%Y%m%d_%H%M%S", time_struct)
            with log_file.open("a") as f:
                f.write(f"{time_str} - Inserted at position ({avg_x:.2f}, {avg_y:.2f}, {avg_yaw:.2f})\n")
                f.write(f"Caption: {caption_text}\n")

        self.memory.insert(entity)
        print(f"Inserted at position ({avg_x:.2f}, {avg_y:.2f}, {avg_yaw:.2f})\n")