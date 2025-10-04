# remote_captioner.py
# A drop-in Captioner that calls a remote (OpenAI-compatible) multimodal API.
# Works with vLLM's OpenAI server or any OpenAI-compatible endpoint that supports images.

import argparse
import base64
import os
import re
from io import BytesIO
from typing import List

import requests
from PIL import Image

from remembr.captioners.captioner import Captioner

# DEFAULT_QUERY = (
#     "<video> Please describe in detail what you see in the few seconds of the video. "
#     "Specifically focus on the people, objects, environmental features, events/activities, "
#     "and other interesting details. Think step by step about these details and be very specific."
# )

DEFAULT_QUERY = ("Please describe what you see in the few seconds of the video.")


def image_parser(args):
    # keep original CLI behavior from your file
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        resp = requests.get(image_file, timeout=30)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files: List[str]):
    return [load_image(p) for p in image_files]


def _pil_to_data_url(img: Image.Image, fmt="PNG") -> str:
    """Encode a PIL image as data URL (base64) for Chat Completions image_url usage."""
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


class RemoteAPICaptioner(Captioner):
    """
    Remote multimodal captioner that sends images to an OpenAI-compatible
    /v1/chat/completions endpoint (e.g., vLLM OpenAI server).

    Required args:
      - api_base:   e.g. http://localhost:8000/v1  (or https://your.domain/v1)
      - model:      remote model name served by the API (e.g. 'Qwen/Qwen2-VL-7B-Instruct')
      - api_key:    (optional) bearer token if your server requires it (also reads env OPENAI_API_KEY)
      - query:      user text prompt for captioning
      - temperature, top_p, max_new_tokens: decoding params (optional)
      - system_prompt: (optional) system message; defaults to a concise captioning instruction

    This class intentionally has no dependency on llava/vila/torch/etc.
    """

    def __init__(self, api_base="http://localhost:11434/v1", model_type="qwen2.5vl:7b", args=None):
        self.api_base = api_base
        self.model = model_type

        # # TODO
        # self.api_base = "https://api.siliconflow.cn/v1/"
        # self.model = "Pro/Qwen/Qwen2.5-VL-7B-Instruct"
        
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.timeout = 600
        self.temperature = 0.2
        self.top_p = 0.9
        self.max_new_tokens = 128
        self.query = DEFAULT_QUERY
        self.system_prompt = "You are a helpful vision assistant. Describe the image(s) clearly, factual, concise."

        print("Use RemoteAPICaptioner:")
        print("\tAPI base:", self.api_base)
        print("\tModel:", self.model)

        self.api_base = self.api_base.rstrip("/")

        # allow passing api_key via args or environment var
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY", "")

        # Optional: support placeholder replacement like <image> in text,
        # though OpenAI-style APIs don't require it. We simply ignore it.
        self.image_placeholder_pattern = re.compile(r"<image>|<image_\d+>")

    def _build_messages(self, images: List[Image.Image]):
        """
        Builds messages for OpenAI-compatible chat.completions with image_url (data URLs).
        Puts all images into a single user message alongside the text query.
        """
        # Encode images as data URLs
        image_contents = [{"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}} for img in images]

        # Clean user query: remove legacy placeholders if present
        user_text = self.image_placeholder_pattern.sub("", self.query or "").strip()
        if not user_text:
            user_text = "Please describe what you see in the few seconds of the video. Especially focus on objects and their attributes, like trash can in black, fire hydrant, etc."

        user_content = [{"type": "text", "text": user_text}] + image_contents

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        return messages

    def _headers(self):
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def caption(self, images: List[Image.Image]):
        """
        Sends a single chat.completions request and returns the assistant content.
        """
        # print("[Debug]: Sending request to remote API...")
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": self._build_messages(images),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
        }

        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        # OpenAI/vLLM-compatible shape: choices[0].message.content
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # Provide debugging info without crashing the caller
            return f"[RemoteAPICaptioner] Unexpected response: {data}"

if __name__ == "__main__":
    args = argparse.ArgumentParser("Remote multimodal captioner (OpenAI/vLLM compatible)")
    args.add_argument("--image-file", type=str, required=False, default="", help="Single path or sep-joined list")
    args.add_argument("--sep", type=str, default=",", help="Separator for multiple image paths")
    args = args.parse_args() if isinstance(args, argparse.ArgumentParser) else args
    cap = RemoteAPICaptioner()
    paths = image_parser(args) if args.image_file else []
    images = load_images(paths) if paths else []
    out = cap.caption(images)
    print(out)
