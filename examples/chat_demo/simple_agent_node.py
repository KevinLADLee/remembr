import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler
import re


class SimpleAgentNode(Node):
    def __init__(self):
        super().__init__('simple_agent_node')
        self.goal_publisher = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )
        self.get_logger().info('Simple Agent Node initialized')

    def send_goal(self, x, y, yaw):
        """Send a navigation goal to Nav2 via topic"""
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = 0.0

        # Convert yaw to quaternion
        q = quaternion_from_euler(0, 0, yaw)
        goal_msg.pose.orientation.x = q[0]
        goal_msg.pose.orientation.y = q[1]
        goal_msg.pose.orientation.z = q[2]
        goal_msg.pose.orientation.w = q[3]

        self.get_logger().info(f'Publishing goal: x={x}, y={y}, yaw={yaw}')
        self.goal_publisher.publish(goal_msg)

    def parse_and_send_goal(self, response_text):
        """Parse response text for [x, y, yaw] pattern and send goal"""
        # Pattern to match [x, y, yaw] format
        pattern = r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]'
        match = re.search(pattern, response_text)

        self.get_logger().info(f'Parsing response: {response_text}')

        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            yaw = float(match.group(3))

            self.get_logger().info(f'Parsed coordinates: x={x}, y={y}, yaw={yaw}')
            self.send_goal(x, y, yaw)
            return True
        else:
            self.get_logger().info('No navigation coordinates found in response')
            return False
