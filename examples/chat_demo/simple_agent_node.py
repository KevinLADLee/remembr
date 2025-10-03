import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler
import re
import threading


class SimpleAgentNode(Node):
    def __init__(self):
        super().__init__('simple_agent_node')
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info('Simple Agent Node initialized')

    def send_goal(self, x, y, yaw):
        """Send a navigation goal to Nav2"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Convert yaw to quaternion
        q = quaternion_from_euler(0, 0, yaw)
        goal_msg.pose.pose.orientation.x = q[0]
        goal_msg.pose.pose.orientation.y = q[1]
        goal_msg.pose.pose.orientation.z = q[2]
        goal_msg.pose.pose.orientation.w = q[3]

        self.get_logger().info(f'Waiting for Nav2 action server...')
        self._action_client.wait_for_server()

        self.get_logger().info(f'Sending goal: x={x}, y={y}, yaw={yaw}')
        send_goal_future = self._action_client.send_goal_async(goal_msg)

        return send_goal_future

    def parse_and_send_goal(self, response_text):
        """Parse response text for [x, y, yaw] pattern and send goal"""
        # Pattern to match [x, y, yaw] format
        pattern = r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]'
        match = re.search(pattern, response_text)

        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            yaw = float(match.group(3))

            self.get_logger().info(f'Parsed coordinates: x={x}, y={y}, yaw={yaw}')

            future = self.send_goal(x, y, yaw)
            rclpy.spin_until_future_complete(self, future)

            goal_handle = future.result()
            if goal_handle.accepted:
                self.get_logger().info('Navigation goal accepted!')
                return True
            else:
                self.get_logger().warn('Navigation goal rejected!')
                return False
        else:
            self.get_logger().info('No navigation coordinates found in response')
            return False
