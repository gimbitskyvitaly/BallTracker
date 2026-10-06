"""Ball detection inference ported from
https://github.com/asigatchov/fast-volleyball-tracking-inference

The model family, preprocessing (grayscale + resize to 512x288, float32/255),
sequential ONNX inference with a rolling frame buffer and heatmap post-processing
(contour moments -> ball center) are taken from
``src/inference_onnx_seq_gray_v2.py`` of that repository.  Only the parts needed
for the pass-trajectory service are kept here.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

LOG = logging.getLogger(__name__)

DEFAULT_INPUT_WIDTH = 512
DEFAULT_INPUT_HEIGHT = 288
DEFAULT_HEATMAP_THRESHOLD = 0.5

BALL_RADIUS_MIN = 3
BALL_RADIUS_MAX = 40


def infer_model_params(model_path: str) -> dict:
    """Model metadata inferred from the file name (same logic as upstream)."""
    model_name = os.path.basename(model_path).lower()
    if "vballnetgrid" in model_name:
        return {
            "family": "grid",
            "seq": 9,
            "input_seq": 9,
            "input_width": 768,
            "input_height": 432,
            "grid_cols": 48,
            "grid_rows": 27,
            "planes": 3,
        }
    return {
        "family": "heatmap",
        "seq": 15 if "seq15" in model_name else 9 if "seq9" in model_name else 3,
        "input_seq": 15 if "seq15" in model_name else 9,
        "input_width": DEFAULT_INPUT_WIDTH,
        "input_height": DEFAULT_INPUT_HEIGHT,
        "grid_cols": None,
        "grid_rows": None,
        "planes": 1,
    }


class BallDetector:
    """Sequential VballNet ball detector running on ONNX Runtime."""

    def __init__(self, model_path: str, device: str = "CPU") -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.params = infer_model_params(model_path)
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.upper() == "CUDA"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path, providers=providers)

        input_names = [i.name for i in self.session.get_inputs()]
        self.input_names = input_names
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.has_gru = "h0" in input_names
        self.batch_size = self.params["input_seq"]

        self.h0 = None
        if self.has_gru:
            for inp in self.session.get_inputs():
                if inp.name == "h0":
                    shape = []
                    for dim in inp.shape:
                        if isinstance(dim, str) or dim is None:
                            shape.append(512 if "hidden" in str(dim).lower() else 1)
                        else:
                            shape.append(dim)
                    self.h0 = np.zeros(tuple(shape), dtype=np.float32)

        out_shape = self.session.get_outputs()[0].shape
        out_channels = out_shape[1] if isinstance(out_shape[1], int) else self.batch_size
        if self.params["family"] == "grid":
            planes = 4 if out_channels % 4 == 0 and out_channels // 4 == self.batch_size else 3
            self.out_dim = out_channels // planes
        else:
            planes = 2 if out_channels == self.batch_size * 2 else 1
            self.out_dim = out_channels // planes
        self.params["planes"] = planes
        self.params["seq"] = self.out_dim

        self.frame_buffer: List[np.ndarray] = []
        LOG.info(
            "Loaded ball model %s (family=%s, seq=%s, gru=%s)",
            os.path.basename(model_path),
            self.params["family"],
            self.out_dim,
            self.has_gru,
        )

    # ------------------------------------------------------------ pre/post
    @staticmethod
    def preprocess(frame: np.ndarray, height: int, width: int) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (width, height))
        return small.astype(np.float32) / 255.0

    def postprocess_heatmap(self, output: np.ndarray, threshold: float) -> List[Tuple[int, int, int]]:
        """Decode heatmaps into (visibility, x, y) in *model input* coordinates."""
        results: List[Tuple[int, int, int]] = []
        h = self.params["input_height"]
        w = self.params["input_width"]
        for idx in range(self.out_dim):
            heatmap = output[0, idx, :, :]
            _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                moments = cv2.moments(largest)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    results.append((1, cx, cy))
                    continue
            results.append((0, 0, 0))
        return results

    def postprocess_grid(self, output: np.ndarray, threshold: float) -> List[Tuple[int, int, int]]:
        seq = self.out_dim
        rows = self.params["grid_rows"]
        cols = self.params["grid_cols"]
        planes = self.params["planes"]
        w = self.params["input_width"]
        h = self.params["input_height"]
        reshaped = output[0].reshape(seq, planes, rows, cols)
        results: List[Tuple[int, int, int]] = []
        for idx in range(seq):
            conf = reshaped[idx, 0]
            max_index = int(np.argmax(conf))
            row = max_index // cols
            col = max_index % cols
            if float(conf[row, col]) < threshold:
                results.append((0, 0, 0))
                continue
            x = (col + float(reshaped[idx, 1, row, col])) * (w / cols)
            y = (row + float(reshaped[idx, 2, row, col])) * (h / rows)
            results.append((1, int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))))
        return results

    def decode(self, output: np.ndarray, threshold: float) -> List[Tuple[int, int, int]]:
        if self.params["family"] == "grid":
            return self.postprocess_grid(output, threshold)
        return self.postprocess_heatmap(output, threshold)

    # ------------------------------------------------------------- inference
    def detect_batch(
        self, frames: List[np.ndarray], threshold: float
    ) -> List[Optional[Tuple[float, float]]]:
        """Run one sequential inference; return per-frame (x, y) in source px.

        Coordinates are scaled back from model input size to the source frame.
        ``None`` marks an invisible ball.
        """
        h_in = self.params["input_height"]
        w_in = self.params["input_width"]
        processed = [self.preprocess(f, h_in, w_in) for f in frames]

        while len(self.frame_buffer) < self.batch_size:
            self.frame_buffer.append(
                processed[0] if processed else np.zeros((h_in, w_in), dtype=np.float32)
            )
        for pf in processed:
            self.frame_buffer.append(pf)
        self.frame_buffer = self.frame_buffer[-self.batch_size:]

        tensor = np.stack(self.frame_buffer, axis=2)[None, :, :, :]
        tensor = np.transpose(tensor, (0, 3, 1, 2)).astype(np.float32)

        inputs = {self.input_names[0]: tensor}
        feed_h0 = self.has_gru and self.h0 is not None
        if feed_h0:
            inputs[self.input_names[1]] = self.h0

        outputs = self.session.run(self.output_names, inputs)
        if feed_h0 and len(outputs) > 1:
            self.h0 = outputs[1]

        raw = outputs[0]
        # tf2onnx-exported VballNet heatmaps may come out in logit space;
        # apply sigmoid so the heatmap threshold stays in [0, 1].
        if raw.min() < -0.01 or raw.max() > 1.01:
            raw = 1.0 / (1.0 + np.exp(-np.clip(raw, -60.0, 60.0)))
        predictions = self.decode(raw, threshold)

        src_w = frames[0].shape[1]
        src_h = frames[0].shape[0]
        results: List[Optional[Tuple[float, float]]] = []
        for visibility, x, y in predictions[: len(frames)]:
            if visibility:
                results.append((x * src_w / w_in, y * src_h / h_in))
            else:
                results.append(None)
        return results
