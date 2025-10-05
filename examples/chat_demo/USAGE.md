# ReMEmbR Chat Demo

✨ A Gradio-based showcase for the ReMEmbR Agent backed by Milvus vector search. Import ROS2 bag files or raw datasets, converse with the agent, and optionally forward detected navigation goals to Nav2.

## 🧩 Installation & Setup

1. 🐳 Install Docker CE following the [official guide](https://docs.docker.com/engine/install/ubuntu/) (or the [China mirror](https://developer.aliyun.com/mirror/docker-ce?spm=a2c6h.13651102.0.0.29c01b11kPgoeM)).
2. 📦 Create a Python virtual environment (`venv`, `conda`, `uv`, …).
3. 🧪 Install project dependencies; pre-installing PyTorch is recommended (see the [PyTorch setup guide](https://pytorch.org/get-started/locally/)).
   ```bash
   pip install -r requirements.txt
   ```
4. 🧠 Configure Ollama if you plan to build the memory module (example: `qwen2.5vl:7b`).
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh

   # Pull the model
   ollama pull qwen2.5vl:7b
   ```
5. 🔧 Install ReMEmbR in editable mode.
   ```bash
   pip install -e .
   ```

## 🚀 Run the Demo

1. 🗄️ Start Milvus (default version 2.5.4) to store memories.
   ```bash
   bash standalone_embed.sh start
   ```
2. 🌐 Launch the Gradio interface.
   ```bash
   # Use a mirror if Hugging Face is unreachable
   export HF_ENDPOINT=https://hf-mirror.com

   # Provide an API key for the selected LLM backend (empty if not needed)
   export OPENAI_API_KEY="sk-628d9161a07c40cfaac35a3a9e9f096e"

   # Start the chat demo
   python -u examples/chat_demo/demo.py --llm_backend qwen3-235b-a22b
   ```

**⚙️ Key flags**
- `--llm_backend`: LLM backend identifier for the ReMEmbR Agent.
- `--rosbag_enabled`: Enables ROS2 bag ingestion UI; requires ROS2 Humble+ and the `ros2 bag` CLI.
- `--enable_ros2_nav`: Requires a robot-side [Nav2](https://docs.nav2.org/getting_started/index.html) stack; the chatbot publishes goals to `/goal_pose`.
- `--chatbot_host_ip`: Host interface Gradio binds to (default `0.0.0.0`).
- `--chatbot_host_port`: Gradio port (default `7860`).
- `--db_uri`: Milvus connection string; ingestion only uses the host portion.

## 💬 Using the UI

1. **📁 Configure Milvus**
   - Ensure `Database URI` targets a reachable Milvus instance.
   - Click `Refresh Collections` to list available collections.
   - Click `Set Collection` to bind memory; the message input unlocks afterward.
2. **🤖 Chat with the agent**
   - Submit messages via Enter; the `Inference log` shows streaming graph output.
   - When Nav2 is active, parsed `[x, y, yaw]` poses appear under `Goal Pose`.
   - `Clear` wipes the chat, input box, and log area.
3. **🧾 Inspect logs**
   - `Inference log` records node-by-node graph output for debugging and tracing.

## 📥 Data Ingestion Options

### 📂 Raw dataset mode (default)
1. Enter a valid collection name in `Name of new DB collection`.
2. Provide the dataset folder path compatible with `RawDbMemoryBuilder`.
3. Once validation passes, click `Process Raw Dataset` to start ingestion.

### 👜 ROS2 bag mode (`--rosbag_enabled`)
1. Upload a `.db3` ROS2 bag file.
2. Choose the desired position topics (e.g., `/odom`, `/amcl_pose`) and image topics.
3. Supply a valid target collection name; invalid names trigger inline errors.
4. Click `Start Processing`; the helper launches the builder and runs `ros2 bag play`.
5. Processing stops after playback completes; check the terminal for detailed logs.

## 🛰️ Nav2 Goal Forwarding (`--enable_ros2_nav`)
- Requires a running Nav2 stack and an importable `SimpleAgentNode` implementation.
- When responses include `[x, y, yaw]`, the pose displays in `Goal Pose`.
- `Send Goal` activates with a detected pose and forwards it via `SimpleAgentNode.send_goal`.

## 🛠️ Troubleshooting
- **✅ Collection naming**: Must start with a letter/underscore and contain only letters, digits, or underscores.
- **🔌 Milvus connectivity**: If `<failed to load>` appears, verify `db_uri` and Milvus service status.
- **🦾 ROS2 environment**: Source the appropriate ROS2 setup so `ros2` CLI and message types resolve.
- **📚 Imports**: Run from the repo root (or adjust `PYTHONPATH`) so modules like `remembr.agents.remembr_agent` load correctly.

## 📑 Raw Dataset Guidelines

See `preprocess/rosbagh2_to_dataset.py` for a detailed reference. Example layout:

```
raw_dataset
├── imgs
│   └── 1755955789.392826112.jpg
└── poses.csv
```

- Image filenames should be timestamped.
- `poses.csv` should follow the schema below (currently only `yaw` is consumed; `roll`, `pitch`, `z` are reserved).

| timestamp | x | y | z | roll | pitch | yaw |
| --- | --- | --- | --- | --- | --- | --- |
| 1755955789.392826112 | -1.11278661590906 | -0.1237228769044 | 0 | 0 | 0 | -0.0030477609763441 |
