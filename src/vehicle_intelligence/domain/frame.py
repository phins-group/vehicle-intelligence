from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class VideoFrame:
    camera_id: str
    frame_id: int
    timestamp: datetime
    image: NDArray[np.uint8]
    stream_epoch: int = 0

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id cannot be negative")
        if self.stream_epoch < 0:
            raise ValueError("stream_epoch cannot be negative")
        if self.timestamp.tzinfo is None:
            raise ValueError("frame timestamp must be timezone-aware")
        if self.image.ndim not in (2, 3) or self.image.size == 0:
            raise ValueError("frame image must be a non-empty grayscale or color array")
