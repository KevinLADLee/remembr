from dataclasses import dataclass

from PIL import Image

from remembr.memory.memory import Memory, MemoryItem

FIXED_SUBTRACT=1721761000 # this is just a large value that brings us closed to 1970


@dataclass
class ImageMemoryItem(MemoryItem):
    time: float
    position: list
    theta: float
    image: Image.Image


class VideoMemory(Memory):

    def __init__(self, fps=1):
        self.memory = []
        self.last_memory_time = 0
        self.fps = fps

    def insert(self, item: ImageMemoryItem):
        self.memory.append(item)

        # current_time = item.time

        # if self.last_memory_time - current_time > 1/self.fps:
        #     self.last_memory_time

    def reset(self):
        self.memory = []

    def get_working_memory(self) -> list[ImageMemoryItem]:
        return self.memory



def format_memory(memory_item_list: list[MemoryItem]):
    return memory_item_list
