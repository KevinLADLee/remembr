<h1 align="center">ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robots</h1>

<p align="center"><a href="https://arxiv.org/abs/2409.13682">Paper</a> - <a href="https://nvidia-ai-iot.github.io/remembr">Website</a><br><a href="#setup">Setup</a> -<a href="#usage">Usage</a> - <a href="#examples">Examples</a> - <a href="#evaluation">Evaluation</a> - <a href="#notice">Notice</a> - <a href="#see_also">See also</a> 

</p>


ReMEmbR is a project that uses LLMs + VLMs to build and reason over
long horizon spatio-temporal memories.  

This allows robots to reason over many kinds of questions such as 
"Hey Robot, can you take take me to get snacks?" to temporal multi-step reasoning 
questions like "How long were you in the building for?"

<a id="setup"></a>
## Setup

1. Clone the repository

    ```
    https://github.com/KevinLADLee/remembr.git
    # create virtual env, and install depends
    pip3 install -r requirements.txt    
    ```

2. Install OLLama

    ```
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull qwen2.5vl:7b
    ```

3. Install MilvusDB

    > `docker` must be installed on the system to easily use Milvus by simply running the command below. This script will automatically launch MilvusDB on a docker container. Otherwise, the user must install MilvusDB from scratch themselves.

    https://milvus.io/docs/install_standalone-docker.md

    ```
    bash standalone_embed.sh start
    ```

## Runtime Setup

You should setup your api keys and hugging face mirror (if need)
```bash
    export HF_ENDPOINT=https://hf-mirror.com
    export OPENAI_API_KEY=""
```

## Examples

### Example 1 - ROS Bag Gradio Demo (offline)

1. Follow the setup above

2. Run the demo

    ```bash
    cd examples/chat_demo
    python -u examples/chat_demo/demo.py --rosbag_enabled --llm_backend qwen3-235b-a22b
    ```
3. Open your web browser to load your ROSBag and query the agent

if you need migration with other LLM api, check details in `remembr/agents/remembr_agent.py`.


### Example 2 - Nova Carter Demo (live)

Please check the [nova_carter_demo](./examples/nova_carter_demo) folder for details.


