"""
Gradio-based chat demo for ReMEmbR agent with Milvus memory backend.
Supports ROS2 bag processing and raw dataset ingestion.
"""
import sys
sys.path.append(sys.path[0] + '/..')

import argparse
import subprocess
import threading
from typing import List, Dict, Any

import gradio as gr
import torch
from pymilvus import MilvusClient

from remembr.memory.milvus_memory import MilvusMemory, ensure_event_loop
from remembr.agents.remembr_agent import ReMEmbRAgent

torch.multiprocessing.set_start_method('spawn')


class StoppableThread(threading.Thread):
    """Thread with graceful stop capability."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()

    def stopped(self):
        """Check if stop has been signaled."""
        return self._stop_event.is_set()


class GradioDemo:
    """Main Gradio demo application for ReMEmbR agent."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.agent = ReMEmbRAgent(llm_type=args.llm_backend)
        self.db_dict: Dict[str, Any] = {}
        self.nav_agent = None
        self.collection_is_set = False

        # Initialize ROS2 if needed
        if args.rosbag_enabled or args.enable_ros2_nav:
            import rclpy
            rclpy.init()

        # Set mode flags
        self.rosbag_enabled = args.rosbag_enabled
        self.raw_dataset_enabled = not args.rosbag_enabled
        self.enable_ros2_nav = args.enable_ros2_nav

        # Initialize Nav2 agent if enabled
        if args.enable_ros2_nav:
            self._init_nav_agent()

        self.launch_demo()

    def _init_nav_agent(self):
        """Initialize the Nav2 simple agent node."""
        try:
            from simple_agent_node import SimpleAgentNode
            self.nav_agent = SimpleAgentNode()
            print("Nav2 Simple Agent Node initialized")
        except Exception as e:
            print(f"Failed to initialize Nav2 agent: {e}")
            self.nav_agent = None

    # === Database Operations ===

    def _list_collections(self, db_uri: str) -> List[str]:
        """List all available collections in the Milvus database."""
        ensure_event_loop()
        client = MilvusClient(uri=db_uri)
        return client.list_collections()

    def update_remembr(self, db_uri: str, collection_name: str):
        """Update the agent's memory to use the specified collection."""
        ensure_event_loop()
        db_ip = self._extract_db_ip(db_uri)
        memory = MilvusMemory(collection_name, db_ip=db_ip)
        self.agent.set_memory(memory)
        self.collection_is_set = True
        print(f"Set collection to: {collection_name}")

    def _extract_db_ip(self, db_uri: str) -> str:
        """Extract IP address from database URI."""
        return db_uri.split('://')[1].split(':')[0]

    # === Dataset Processing ===

    def process_raw_dataset(self, dataset_path: str, collection_name: str, db_uri: str):
        """Process and ingest a raw dataset into Milvus."""
        from raw_db_processor import RawDbMemoryBuilder

        print(f"Processing raw dataset: {dataset_path}")

        self.db_dict['collection_name'] = collection_name
        self.db_dict['db_ip'] = self._extract_db_ip(db_uri)
        self.db_dict['dataset_path'] = dataset_path

        mem_builder = RawDbMemoryBuilder(
            self.db_dict['dataset_path'],
            collection_name=self.db_dict['collection_name'],
            db_ip=self.db_dict['db_ip']
        )
        mem_builder.run()
        print("Done processing raw dataset")

    def process_file(self, fileobj, collection_name: str, pos_topic: str,
                     image_topic: str, db_uri: str):
        """Process a ROS2 bag file and ingest into Milvus."""
        from ros2_db_processor import create_and_launch_memory_builder, validate_rosbag_topics

        print(f"Processing ROS2 bag file: {fileobj.name}")

        # Validate collection name
        if not collection_name or not collection_name.strip():
            error_msg = "Error: Collection name is required"
            print(error_msg)
            raise gr.Error(error_msg)

        # Validate topics in bag file
        is_valid, message, topics_dict = validate_rosbag_topics(
            fileobj.name, pos_topic, image_topic
        )

        if not is_valid:
            error_msg = f"{message}\nAvailable position topics: {topics_dict['position']}\nAvailable image topics: {topics_dict['image']}"
            print(error_msg)
            raise gr.Error(error_msg)

        print(f"Topic validation passed: {message}")

        self.db_dict['collection_name'] = collection_name
        self.db_dict['pos_topic'] = pos_topic
        self.db_dict['image_topic'] = image_topic
        self.db_dict['db_ip'] = self._extract_db_ip(db_uri)

        # Create memory builder
        mem_builder = lambda: create_and_launch_memory_builder(
            None,
            db_ip=self.db_dict['db_ip'],
            collection_name=self.db_dict['collection_name'],
            pos_topic=self.db_dict['pos_topic'],
            image_topic=self.db_dict['image_topic']
        )

        # Launch processing thread
        proc = StoppableThread(target=mem_builder)
        proc.start()

        # Play ROS bag
        bag_process = subprocess.Popen(
            ["ros2", "bag", "play", fileobj.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )

        # Wait for bag playback to complete
        while bag_process.poll() is None:
            pass

        print(f"ROS2 bag playback completed with return code: {bag_process.returncode}")
        proc.stop()

    def _parse_bag_topics(self, fileobj):
        """Parse topics from uploaded ROS2 bag file."""
        from ros2_db_processor import get_rosbag_topics

        if fileobj is None:
            return gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(visible=False)

        try:
            topics_dict = get_rosbag_topics(fileobj.name)
            pos_topics = topics_dict['position']
            img_topics = topics_dict['image']

            if not pos_topics or not img_topics:
                msg = []
                if not pos_topics:
                    msg.append("No position topics (Odometry/PoseWithCovarianceStamped) found")
                if not img_topics:
                    msg.append("No image topics found")
                raise gr.Error("; ".join(msg))

            # Set default values if available
            pos_default = pos_topics[0] if pos_topics else None
            img_default = img_topics[0] if img_topics else None

            return (
                gr.update(choices=pos_topics, value=pos_default, visible=True),
                gr.update(choices=img_topics, value=img_default, visible=True),
                gr.update(visible=True)
            )

        except Exception as e:
            raise gr.Error(f"Failed to parse bag file: {str(e)}")

    # === Chat Handling ===

    def _process_agent_stream(self, user_message: str):
        """Process user message through the agent graph and yield log output."""
        messages = [("user", user_message)]
        inputs = {"messages": messages}
        graph = self.agent.graph
        log_lines = []

        for output in graph.stream(inputs):
            for key, value in output.items():
                log_lines.append("------------")
                log_lines.append(f"Output from node '{key}':")
                for item in value["messages"]:
                    if isinstance(item, tuple):
                        log_lines.append(item[1])
                    else:
                        content = item if isinstance(item, str) else item.content
                        log_lines.append(content)
                    log_lines.append("\n")
            yield output, "\n".join(log_lines)

    def _send_nav_goal_async(self, response_text: str):
        """Send navigation goal in background thread if nav is enabled."""
        if self.enable_ros2_nav and self.nav_agent:
            def send_goal():
                self.nav_agent.parse_and_send_goal(response_text)
            nav_thread = threading.Thread(target=send_goal, daemon=True)
            nav_thread.start()

    def _create_chat_handler(self):
        """Create the chat message handler function."""
        def chatter(user_message, history, log_text):
            """
            Chat handler for Gradio Chatbot with type='messages'.
            Args:
                user_message: User's input message
                history: Chat history as list of dicts with 'role' and 'content'
                log_text: Current log text
            Yields:
                Tuple of (input_box, updated_history, updated_log)
            """
            history = history or []

            # Show user message with placeholder assistant response
            temp_history = history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "..."},
            ]
            yield "", temp_history, ""

            # Process through agent
            final_output = None
            final_log = ""
            for output, log in self._process_agent_stream(user_message):
                final_output = output
                final_log = log
                yield "", temp_history, log

            # Extract final response
            response_msg = final_output["generate"]["messages"][-1]
            out_dict = eval(response_msg)
            response_text = out_dict["text"]

            # Send navigation goal if enabled
            self._send_nav_goal_async(response_text)

            # Update chat history with final response
            chat_history = history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response_text},
            ]
            yield "", chat_history, final_log

        return chatter

    # === UI Layout ===

    def launch_demo(self):
        """Build and launch the Gradio demo interface."""
        chatter = self._create_chat_handler()

        with gr.Blocks() as demo:
            # Build chat section and get components
            chatbot, msg, output_log, db_uri_box, selector = self._build_chat_section(chatter)

            # Build upload section with db_uri_box reference
            self._build_upload_section(db_uri_box)

            demo.queue()
            demo.launch(
                server_name=self.args.chatbot_host_ip,
                server_port=self.args.chatbot_host_port
            )

    def _build_chat_section(self, chatter):
        """Build the main chat interface section and return key components."""
        with gr.Row():
            # Chat column
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(type="messages", label="Chat", height=500)
                msg = gr.Textbox(
                    label="Message Input",
                    placeholder="Please set a collection first...",
                    interactive=False
                )
                clear = gr.Button("Clear")

            # Log column
            with gr.Column(scale=1):
                output_log = gr.Textbox(label="Inference log", lines=16)

            # Database control column
            with gr.Column(scale=1):
                db_uri_box = gr.Textbox(label="Database URI", value=self.args.db_uri)
                selector = gr.Dropdown(choices=[], label="Select Collection", interactive=True)
                with gr.Row():
                    refresh = gr.Button("Refresh Collections")
                    set_btn = gr.Button("Set Collection")

        # Event handlers
        msg.submit(chatter, [msg, chatbot, output_log], [msg, chatbot, output_log])
        clear.click(lambda: ([], "", ""), outputs=[chatbot, msg, output_log])

        # Collection management
        def update_collections(db_uri):
            try:
                return gr.update(choices=self._list_collections(db_uri))
            except Exception as e:
                return gr.update(choices=["<failed to load>"])

        def set_collection_and_enable(db_uri, collection_name):
            """Set collection and enable message input."""
            self.update_remembr(db_uri, collection_name)
            return gr.update(interactive=True, placeholder="Type your message and press Enter")

        refresh.click(update_collections, inputs=[db_uri_box], outputs=[selector])
        set_btn.click(
            set_collection_and_enable,
            inputs=[db_uri_box, selector],
            outputs=[msg]
        )

        return chatbot, msg, output_log, db_uri_box, selector

    def _build_upload_section(self, db_uri_box):
        """Build the dataset upload section based on enabled mode."""
        if self.rosbag_enabled:
            self._build_rosbag_upload(db_uri_box)
        elif self.raw_dataset_enabled:
            self._build_raw_dataset_upload(db_uri_box)
        else:
            gr.Markdown("**ROS2 is not enabled, file upload disabled.**")

    def _build_rosbag_upload(self, db_uri_box):
        """Build ROS2 bag file upload interface."""
        with gr.Row():
            with gr.Column(scale=1):
                file_upload = gr.File(label="ROS2 Bag File")

                # Topic selection (initially hidden)
                pos_topic = gr.Dropdown(
                    choices=[],
                    label="Position Topic",
                    visible=False
                )
                image_topic = gr.Dropdown(
                    choices=[],
                    label="Image Topic",
                    visible=False
                )

                # Configuration section (initially hidden)
                with gr.Group(visible=False) as config_group:
                    upload_name = gr.Textbox(label="Name of new DB collection")
                    start_btn = gr.Button("Start Processing", variant="primary", interactive=False)

                # Upload triggers topic parsing
                file_upload.upload(
                    self._parse_bag_topics,
                    inputs=[file_upload],
                    outputs=[pos_topic, image_topic, config_group]
                )

                # Enable start button when collection name is filled
                def enable_start(collection_name):
                    is_filled = collection_name and collection_name.strip()
                    return gr.update(interactive=is_filled)

                upload_name.change(
                    enable_start,
                    inputs=[upload_name],
                    outputs=[start_btn]
                )

                # Start processing
                start_btn.click(
                    self.process_file,
                    inputs=[file_upload, upload_name, pos_topic, image_topic, db_uri_box],
                    outputs=[]
                )

    def _build_raw_dataset_upload(self, db_uri_box):
        """Build raw dataset upload interface."""
        with gr.Row():
            with gr.Column(scale=1):
                upload_name = gr.Textbox(label="Name of new DB collection")
                dataset_path = gr.Textbox(label="Raw Dataset Folder Path")
                submit_btn = gr.Button("Process Raw Dataset", interactive=False)

                def update_submit_state(*inputs):
                    """Enable submit button only when all fields are filled."""
                    all_filled = all(inp and inp.strip() for inp in inputs)
                    return gr.update(interactive=all_filled)

                # Update submit button state on text changes
                for textbox in [upload_name, dataset_path]:
                    textbox.change(
                        update_submit_state,
                        inputs=[upload_name, dataset_path],
                        outputs=[submit_btn]
                    )

                submit_btn.click(
                    self.process_raw_dataset,
                    inputs=[dataset_path, upload_name, db_uri_box],
                    outputs=[]
                )



# === Main Entry Point ===

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch Gradio demo for ReMEmbR agent with Milvus memory"
    )
    parser.add_argument(
        "--db_uri",
        type=str,
        default="http://127.0.0.1:19530",
        help="Milvus database URI"
    )
    parser.add_argument(
        "--chatbot_host_ip",
        type=str,
        default="localhost",
        help="Host IP for Gradio server"
    )
    parser.add_argument(
        "--chatbot_host_port",
        type=int,
        default=7860,
        help="Port for Gradio server"
    )
    parser.add_argument(
        "--llm_backend",
        type=str,
        default='codestral',
        help="LLM backend to use"
    )
    parser.add_argument(
        "--rosbag_enabled",
        action='store_true',
        help="Enable ROS2 bag file processing"
    )
    parser.add_argument(
        "--enable_ros2_nav",
        action='store_true',
        help="Enable ROS2 Nav2 navigation goal sending"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    demo = GradioDemo(args)
