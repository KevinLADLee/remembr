import sys
sys.path.append(sys.path[0] + '/..')

import argparse
import gradio as gr
from pymilvus import MilvusClient
import subprocess
import threading
from typing import List
import re
import math

from remembr.memory.milvus_memory import MilvusMemory, ensure_event_loop
from remembr.agents.remembr_agent import ReMEmbRAgent

import torch
torch.multiprocessing.set_start_method('spawn')


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class GradioDemo:
    def __init__(self, args=None):
        self.args = args

        if args.rosbag_enabled or args.enable_ros2_nav:
            import rclpy
            rclpy.init()

        self.rosbag_enabled = args.rosbag_enabled
        self.raw_dataset_enabled = not args.rosbag_enabled
        self.enable_ros2_nav = args.enable_ros2_nav
        self.agent = ReMEmbRAgent(llm_type=args.llm_backend)
        self.db_dict = {}

        # Initialize Nav2 simple agent node if enabled
        self.nav_agent = None
        if args.enable_ros2_nav:
            self._init_nav_agent()

        self.launch_demo()

    def _init_nav_agent(self):
        """Initialize the Nav2 simple agent node"""
        try:
            from simple_agent_node import SimpleAgentNode
            self.nav_agent = SimpleAgentNode()
            print("Nav2 Simple Agent Node initialized")
        except Exception as e:
            print(f"Failed to initialize Nav2 agent: {e}")
            self.nav_agent = None

    # --- helpers ----------------------------------------------------
    def _list_collections(self, db_uri: str) -> List[str]:
        ensure_event_loop()
        client = MilvusClient(uri=db_uri)
        collections = client.list_collections()
        return collections

    def update_remembr(self, db_uri, selection):
        ensure_event_loop()
        ip = db_uri.split('://')[1].split(':')[0]
        memory = MilvusMemory(selection, db_ip=ip)
        self.agent.set_memory(memory)
        print("Set collection to", selection)

    def process_raw_dataset(self, dataset_path, upload_name, db_uri):
        print("Processing raw dataset", dataset_path)
        from raw_db_processor import RawDbMemoryBuilder
        self.db_dict['collection_name'] = upload_name
        self.db_dict['db_ip'] = db_uri.split('://')[1].split(':')[0]
        self.db_dict['dataset_path'] = dataset_path
        print("launching threading")
        mem_builder = RawDbMemoryBuilder(
            self.db_dict['dataset_path'],
            collection_name=self.db_dict['collection_name'],
            db_ip=self.db_dict['db_ip']
        )
        mem_builder.run()
        print("Done processing raw dataset")



    def process_file(self, fileobj, upload_name, pos_topic, image_topic, db_uri):
        from examples.chat_demo.ros2_db_processor import create_and_launch_memory_builder
        print("Processing file", fileobj.name)
        self.db_dict['collection_name'] = upload_name
        self.db_dict['pos_topic'] = pos_topic
        self.db_dict['image_topic'] = image_topic
        self.db_dict['db_ip'] = db_uri.split('://')[1].split(':')[0]

        print("launching threading")
        mem_builder = lambda: create_and_launch_memory_builder(
            None,
            db_ip=self.db_dict['db_ip'],
            collection_name=self.db_dict['collection_name'],
            pos_topic=self.db_dict['pos_topic'],
            image_topic=self.db_dict['image_topic']
        )

        # launch processing thread
        proc = StoppableThread(target=mem_builder)
        proc.start()

        bag_process = subprocess.Popen(
            ["ros2", "bag", "play", fileobj.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )

        while True:
            ret_code = bag_process.poll()
            if ret_code is not None:
                print(ret_code, "DONE")
                proc.stop()
                break

    # --- UI ---------------------------------------------------------
    def launch_demo(self):
        # define chatter in here
        def chatter(user_message, history, log_text):
            """
            使用 Chatbot(type='messages') 的回调：
            - history: List[{'role': 'user'|'assistant', 'content': str}]
            - 返回：(new_msg_input, new_history, new_log_text)
            """
            history = history or []
            # 先把用户消息  占位的助手消息放入历史
            temp_history = history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "..."},
            ]
            # 第一次刷新：清空输入框，展示占位应答，清空/保留日志
            yield "", temp_history, ""

            # 你的业务推理部分
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
                            if isinstance(item, str):
                                log_lines.append(item)
                            else:
                                log_lines.append(item.content)
                        log_lines.append("\n")
                # 中间流式刷新日志（回答仍保持为 "..."）
                yield "", temp_history, "\n".join(log_lines)

            # 生成最终回答
            response_msg = output["generate"]["messages"][-1]
            out_dict = eval(response_msg)
            response_text = out_dict["text"]

            # Send navigation goal if coordinates are detected and nav is enabled
            if self.enable_ros2_nav and self.nav_agent:
                import threading
                def send_nav_goal():
                    self.nav_agent.parse_and_send_goal(response_text)
                nav_thread = threading.Thread(target=send_nav_goal, daemon=True)
                nav_thread.start()

            chat_history = (history or []) + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response_text},
            ]
            yield "", chat_history, "\n".join(log_lines)

        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(type="messages", label="Chat", height=500)
                    msg = gr.Textbox(label="Message Input", placeholder="Type your message and press Enter")
                    clear = gr.Button("Clear")
                with gr.Column(scale=1):
                    output_log = gr.Textbox(label="Inference log", lines=16)

                with gr.Column(scale=1):
                    db_uri_box = gr.Textbox(label="Database URI", value=self.args.db_uri)
                    selector = gr.Dropdown(choices=[], label="Select Collection", interactive=True)
                    with gr.Row():
                        refresh = gr.Button("Refresh Collections")
                        set_btn = gr.Button("Set Collection")

            # 事件绑定
            msg.submit(chatter, [msg, chatbot, output_log], [msg, chatbot, output_log])

            # 清空按钮：清空聊天与输入框与日志
            clear.click(lambda: ([], "", ""), outputs=[chatbot, msg, output_log])

            # 首次载入时和点击刷新按钮时，更新下拉框选项
            def _update_collections(db_uri):
                try:
                    return gr.update(choices=self._list_collections(db_uri))
                except Exception as e:
                    # 出错时保底：不改变现有 choices，但给个占位
                    return gr.update(choices=["<failed to load>"])

            demo.load(_update_collections, inputs=[db_uri_box], outputs=[selector])
            refresh.click(_update_collections, inputs=[db_uri_box], outputs=[selector])

            # 设置被选中的 collection
            set_btn.click(self.update_remembr, inputs=[db_uri_box, selector], outputs=[])

            # 仅在启用 ROS 时展示上传区
            if self.rosbag_enabled:
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_name = gr.Textbox(label="Name of new DB collection")
                        pos_topic = gr.Textbox(label="Position Topic", value="/amcl_pose")
                        image_topic = gr.Textbox(label="Image Topic", value="/camera/color/image_raw")
                        file_upload = gr.File(label="ROS2 Bag File")
                        file_upload.upload(
                            self.process_file,
                            inputs=[file_upload, upload_name, pos_topic, image_topic, db_uri_box],
                            outputs=[]
                        )
            elif self.raw_dataset_enabled:
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_name = gr.Textbox(label="Name of new DB collection")
                        # pos_topic = gr.Textbox(label="Position Topic", value="/amcl_pose")
                        # image_topic = gr.Textbox(label="Image Topic", value="/camera/color/image_raw")
                        dataset_path = gr.Textbox(label="Raw Dataset Folder Path")
                        submit_btn = gr.Button("Process Raw Dataset", interactive=False)
                        
                        def update_submit_state(*inputs):
                            # Check if all required fields are filled
                            all_filled = all(inp and inp.strip() for inp in inputs)
                            return gr.update(interactive=all_filled)
                        
                        # Update submit button state when any textbox changes
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
            else :
                gr.Markdown("**ROS2 is not enabled, file upload disabled.**")


        # v4：不传参最稳，默认就启用队列与并发控制
        demo.queue()
        demo.launch(server_name=self.args.chatbot_host_ip, server_port=self.args.chatbot_host_port)


# ----------------- main -----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_uri", type=str, default="http://127.0.0.1:19530")
    parser.add_argument("--chatbot_host_ip", type=str, default="localhost")
    parser.add_argument("--chatbot_host_port", type=int, default=7860)
    parser.add_argument("--llm_backend", type=str, default='codestral')
    parser.add_argument("--rosbag_enabled", action='store_true')
    parser.add_argument("--enable_ros2_nav", action='store_true', help='Enable ROS2 Nav2 navigation goal sending')

    args = parser.parse_args()
    demo = GradioDemo(args)
