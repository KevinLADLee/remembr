import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped 
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String
from remembr.memory.milvus_memory import MilvusMemory
from remembr.agents.remembr_agent import ReMEmbRAgent
from scipy.spatial.transform import Rotation as R
import traceback
import numpy as np

from common_utils import format_pose_msg


class AgentNode(Node):

    def __init__(self):
        super().__init__("AgentNode")

        self.declare_parameter("llm_type", "command-r7b")
        self.declare_parameter("db_collection", "test03_gps_oss")
        self.declare_parameter("db_ip", "127.0.0.1")
        self.declare_parameter("query_topic", "/speech")
        self.declare_parameter("pose_topic", "/amcl_pose")
        self.declare_parameter("goal_pose_topic", "/goal_pose")

        # look for "robot" keyword
        self.query_filter = lambda text: "robot" in text.lower()

        self.query_subscriber = self.create_subscription(
            String,
            self.get_parameter("query_topic").value,
            self.query_callback,
            10
        )

        self.pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter("pose_topic").value,
            self.pose_callback,
            10
        )

        self.goal_pose_publisher = self.create_publisher(
            PoseStamped,
            self.get_parameter("goal_pose_topic").value,
            10
        )

        self.memory = MilvusMemory(
            self.get_parameter("db_collection").value,
            self.get_parameter("db_ip").value
        )
        self.agent = ReMEmbRAgent(
            llm_type=self.get_parameter("llm_type").value
        )
        self.agent.set_memory(self.memory)

        self.last_pose = None
        self.logger = self.get_logger()
        

    def query_callback(self, msg: String):
        print("Received query: " + msg.data)
        start_time = self.get_clock().now()
        if not self.query_filter(msg.data):
            self.logger.info("Skipping query {msg.data} because it does not have keyword")
            return 

        try:
            query = msg.data 

            # Add additional context information to query
            if self.last_pose is not None:
                position, angle, current_time = format_pose_msg(self.last_pose)
                query +=  f"\nYou are currently located at {position} and the time is {self.current_time}."

            # Run the Remembr Agent
            response = self.agent.query(query)
            if response is None:
                self.logger.info("No response from agent")
                return

            # Generate the goal pose from the response
            position = response.position
            orientation = response.orientation

            # Check position is a valid 3D coordinate
            if len(position) != 3 or not all(isinstance(x, (int, float)) for x in position):
                self.logger.info("Invalid position in response")
                return
            if len(orientation) != 1 or not isinstance(orientation[0], (int, float)):
                self.logger.info("Invalid orientation in response, set to 0")
                orientation = [0.0]

            quat = R.from_euler('z', orientation).as_quat()
            quat = np.squeeze(quat)
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.pose.position.x = float(position[0])
            goal_pose.pose.position.y = float(position[1])
            goal_pose.pose.position.z = float(position[2])
            goal_pose.pose.orientation.x = float(quat[0])
            goal_pose.pose.orientation.y = float(quat[1])
            goal_pose.pose.orientation.z = float(quat[2])
            goal_pose.pose.orientation.w = float(quat[3])

            # Publish the result
            self.logger.info(f"Query took {(self.get_clock().now() - start_time).nanoseconds / 1e9:.2f} seconds")
            self.logger.info("Query executed: ")
            self.logger.info("\tText: " + response.text)
            self.logger.info(f"\tPosition: {position}")
            self.logger.info(f"\tOrientation: {orientation}")
        
            self.goal_pose_publisher.publish(goal_pose)
        except:
            print("FAILED. Returning")
            print(traceback.format_exc())
            return

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        self.last_pose = msg


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()