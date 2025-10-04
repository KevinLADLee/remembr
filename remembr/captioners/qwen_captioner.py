import atexit
import base64
import signal
import sys
from io import BytesIO
from typing import List

import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from qwen_vl_utils import process_vision_info

from remembr.captioners.captioner import Captioner


DEFAULT_PROMPT = (
    "Briefly describe the objects appearing in the video and their most prominent features, "+ \
    "such as a black trash can, a fire hydrant, lift door and etc. Specifically focus on the objects, environmental " + \
        "features, events/activities, and other interesting details." + \
    "Also briefly describe the entire scene. Keep the entire description as concise as possible."
)

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

class QwenVLCaptioner(Captioner):
    """
    Local Qwen2.5-VL-7B-Instruct captioner that runs the model on CUDA.

    Args:
        model_type: Model identifier for modelscope (default: Qwen/Qwen2.5-VL-7B-Instruct)
        device: Device to run the model on (default: cuda)
        query: Text prompt for captioning (default: DEFAULT_PROMPT)
        max_new_tokens: Maximum tokens to generate (default: 128)
        temperature: Sampling temperature (default: 0.2)
        top_p: Nucleus sampling parameter (default: 0.9)
    """

    def __init__(
        self,
        model_type: str = DEFAULT_MODEL,
        device: str = "cuda",
        query: str = DEFAULT_PROMPT,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        self.model_type = model_type
        self.device = device
        self.query = query
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model = None
        self.processor = None
        self.tokenizer = None

        print("Use QwenVLCaptioner (local):")
        print(f"\tModel: {self.model_type}")
        print(f"\tDevice: {self.device}")
        print(f"\tDefault prompt: {self.query}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_type, torch_dtype="auto"
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_type, max_pixels=max_pixels, min_pixels=min_pixels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)

        # Register cleanup handlers
        atexit.register(self.cleanup)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals (Ctrl+C) gracefully."""
        print("\n[QwenVLCaptioner] Interrupt received, cleaning up...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Release GPU resources and clean up model."""
        try:
            if self.model is not None:
                print("[QwenVLCaptioner] Releasing GPU memory...")
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # Force GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("[QwenVLCaptioner] Cleanup complete")
        except Exception as e:
            print(f"[QwenVLCaptioner] Error during cleanup: {e}")

    def _pil_to_base64(self, img: Image.Image, fmt: str = "PNG") -> str:
        """Convert PIL Image to base64 data URL."""
        buf = BytesIO()
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "image/png" if fmt.upper() == "PNG" else f"image/{fmt.lower()}"
        return f"data:{mime};base64,{b64}"

    def caption(self, images: List[Image.Image]) -> str:
        """
        Generate caption for given images.

        Args:
            images: List of PIL Images to caption

        Returns:
            Generated caption as string
        """
        try:
            # Check if model is still loaded
            if self.model is None:
                return "[QwenVLCaptioner] Error: Model has been cleaned up"

            # Build messages with base64 encoded images
            content = []
            for img in images:
                content.append({
                    "type": "image",
                    "image": self._pil_to_base64(img)
                })
            content.append({"type": "text", "text": self.query})

            messages = [{"role": "user", "content": content}]

            # Process with chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision info to get image inputs
            image_inputs, video_inputs = process_vision_info(messages)

            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            # Decode only the generated part (skip input tokens)
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, outputs)
            ]
            caption = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            return caption
        except KeyboardInterrupt:
            print("\n[QwenVLCaptioner] Caption interrupted by user")
            raise
        except Exception as e:
            import traceback
            return f"[QwenVLCaptioner] Error during captioning: {e}\n{traceback.format_exc()}"

    def __del__(self):
        """Destructor to ensure cleanup when object is destroyed."""
        self.cleanup()