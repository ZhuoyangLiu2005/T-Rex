from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any, Dict


@dataclass
class StreamerOutput:
    """Clean separation of different data types"""

    # Data from Vive tracker
    vive_data: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    source: str = ""


class BaseStreamer(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def reset_status(self):
        pass

    @abstractmethod
    def start_streaming(self):
        pass

    @abstractmethod
    def get(self) -> StreamerOutput:
        """Return StreamerOutput with structured data"""
        pass

    @abstractmethod
    def stop_streaming(self):
        pass
