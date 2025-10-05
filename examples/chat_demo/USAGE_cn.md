# ReMEmbR 聊天演示

✨ 基于 Gradio 的 ReMEmbR Agent 演示界面，使用 Milvus 作为向量检索存储。支持导入 ROS2 bag 文件或原始数据集，并可选地将解析出的导航目标转发给 Nav2。

## 🧩 安装配置

1. 🐳 安装 Docker CE：参考 [官方文档](https://docs.docker.com/engine/install/ubuntu/)（[中国大陆镜像](https://developer.aliyun.com/mirror/docker-ce?spm=a2c6h.13651102.0.0.29c01b11kPgoeM)）。
2. 📦 创建 Python 虚拟环境（推荐 `venv`、`conda`、`uv` 等工具）。
3. 🧪 安装项目依赖，建议提前配置 PyTorch（参考 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)）。
   ```bash
   pip install -r requirements.txt
   ```
4. 🧠 如需构建记忆模块，预先配置 Ollama（以 `qwen2.5vl:7b` 为例）。
   ```bash
   # 安装 Ollama
   curl -fsSL https://ollama.com/install.sh | sh

   # 拉取模型
   ollama pull qwen2.5vl:7b
   ```
5. 🔧 安装 ReMEmbR。
   ```bash
   pip install -e .
   ```

## 🚀 运行流程

1. 🗄️ 启动 Milvus（默认版本 2.5.4），用于存储记忆数据。
   ```bash
   bash standalone_embed.sh start
   ```
2. 🌐 启动 Gradio 演示。
   ```bash
   # 如无法连接 Hugging Face，可使用公益镜像服务
   export HF_ENDPOINT=https://hf-mirror.com

   # 指定后端大模型的 API Key，如无需可设为空字符串
   export OPENAI_API_KEY="sk-628d9161a07c40cfaac35a3a9e9f096e"

   # 运行网页演示
   python -u examples/chat_demo/demo.py --llm_backend qwen3-235b-a22b
   ```

**⚙️ 运行参数速览**
- `--llm_backend`：ReMEmbR Agent 使用的 LLM 后端模型。
- `--rosbag_enabled`：启用后支持上传 ROS2 bag 文件，需 ROS2 Humble 及以上版本，并保证 `ros2 bag` CLI 可用。
- `--enable_ros2_nav`：启用后要求机器人侧支持 [Nav2](https://docs.nav2.org/getting_started/index.html)，Chatbot 会自动发布 `/goal_pose`。
- `--chatbot_host_ip`：Gradio 绑定地址（默认 `0.0.0.0`）。
- `--chatbot_host_port`：Gradio UI 端口（默认 `7860`）。
- `--db_uri`：Milvus 连接字符串，导入数据时使用其中的主机信息。

## 💬 使用 Gradio 界面

1. **📁 设置 Milvus Collection**
   - 确认 `Database URI` 指向可用的 Milvus 服务。
   - 点击 `Refresh Collections` 获取现有 Collection。
   - 点击 `Set Collection` 绑定目标 Collection，聊天输入框随即解锁。
2. **🤖 与 Agent 对话**
   - 输入消息并回车发送，`Inference log` 实时输出推理日志。
   - 如启用 Nav2，解析出的 `[x, y, yaw]` 会显示在 `Goal Pose` 面板。
   - 点击 `Clear` 可清空对话、输入框和日志。
3. **🧾 查看日志**
   - `Inference log` 展示 Agent Graph 的逐节点输出，便于排查或理解响应过程。

## 📥 导入数据

### 📂 原始数据集模式（默认）
1. 在 `Name of new DB collection` 中输入符合规则的 Collection 名称。
2. 填写原始数据集文件夹路径（需符合 `RawDbMemoryBuilder` 的数据格式）。
3. 校验通过后点击 `Process Raw Dataset` 开始导入。

### 👜 ROS2 bag 模式（需 `--rosbag_enabled`）
1. 上传 `.db3` ROS2 bag 文件。
2. 在下拉框中选择位置话题（如 `/odom`、`/amcl_pose` 等）和图像话题。
3. 输入合法的目标 Collection 名称；若不符合规则会即时提示。
4. 点击 `Start Processing`，系统会运行内存构建器并自动执行 `ros2 bag play`。
5. 播放结束后处理自动停止，详细日志可在运行脚本的终端中查看。

## 🛰️ 导航目标发送（需 `--enable_ros2_nav`）
- 依赖已加载的 Nav2 栈以及可导入的 `SimpleAgentNode`。
- 当 Agent 回复中出现形如 `[x, y, yaw]` 的姿态时，会在 `Goal Pose` 面板显示。
- `Send Goal` 在检测到姿态后变为可点击，点击即可通过 `SimpleAgentNode.send_goal` 推送目标点。

## 🛠️ 常见问题排查
- **✅ Collection 名校验**：名称需以字母或下划线开头，仅包含字母、数字和下划线。
- **🔌 Milvus 连接**：若下拉列表显示 `<failed to load>`，请检查 `db_uri` 配置及 Milvus 服务状态。
- **🦾 ROS2 环境**：运行前需 source 对应的 ROS2 环境，确保 `ros2` CLI 与消息类型可用。
- **📚 依赖导入**：脚本默认可以导入 `remembr.agents.remembr_agent` 等模块，建议在项目根目录运行或设置好 `PYTHONPATH`。

## 📑 原始数据集要求

可参考 `preprocess/rosbagh2_to_dataset.py`。文件夹结构示例：

```
raw_dataset
├── imgs
│   └── 1755955789.392826112.jpg
└── poses.csv
```

- 图片文件建议使用时间戳命名。
- `poses.csv` 格式要求如下（当前仅使用 `yaw`，`roll`、`pitch`、`z` 保留备用）：

| timestamp | x | y | z | roll | pitch | yaw |
| --- | --- | --- | --- | --- | --- | --- |
| 1755955789.392826112 | -1.11278661590906 | -0.1237228769044 | 0 | 0 | 0 | -0.0030477609763441 |
