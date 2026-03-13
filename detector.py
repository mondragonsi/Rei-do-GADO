import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Set, List


# COCO dataset class ID for "cow"
COW_CLASS_ID = 19

MODELS = {
    "Nano — Mais Rápido": "yolov8n.pt",
    "Small — Rápido": "yolov8s.pt",
    "Medium — Recomendado ✓": "yolov8m.pt",
    "Large — Alta Precisão": "yolov8l.pt",
    "XLarge — Máxima Precisão": "yolov8x.pt",
}

# Vibrant color palette for individual animal IDs
PALETTE_HEX = [
    "#FF6B35", "#F7C948", "#4CAF50", "#2196F3",
    "#9C27B0", "#F44336", "#00BCD4", "#FF5722",
    "#8BC34A", "#E91E63", "#607D8B", "#FFEB3B",
]


@dataclass
class CattleStats:
    unique_ids: Set[int] = field(default_factory=set)
    max_simultaneous: int = 0
    frame_counts: List[int] = field(default_factory=list)
    total_frames: int = 0

    @property
    def total_unique(self) -> int:
        return len(self.unique_ids)

    @property
    def avg_per_frame(self) -> float:
        if not self.frame_counts:
            return 0.0
        return round(sum(self.frame_counts) / len(self.frame_counts), 1)


class CattleDetector:
    def __init__(self, model_key: str = "Medium — Recomendado ✓", confidence: float = 0.4):
        model_path = MODELS.get(model_key, "yolov8m.pt")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.stats = CattleStats()
        self._init_tracker()
        self._init_annotators()

    def _init_tracker(self):
        self.tracker = sv.ByteTrack()

    def _init_annotators(self):
        try:
            palette = sv.ColorPalette.from_hex(PALETTE_HEX)
        except Exception:
            palette = sv.ColorPalette.DEFAULT

        lookup = sv.ColorLookup.TRACK

        self.box_annotator = sv.BoxAnnotator(
            color=palette,
            color_lookup=lookup,
            thickness=3,
        )
        self.label_annotator = sv.LabelAnnotator(
            color=palette,
            color_lookup=lookup,
            text_color=sv.Color.WHITE,
            text_scale=0.65,
            text_thickness=2,
            text_padding=6,
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=palette,
            color_lookup=lookup,
            thickness=2,
            trace_length=60,
        )

    def reset(self):
        self.stats = CattleStats()
        self._init_tracker()

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Run detection + tracking on a single frame.
        Returns (annotated_frame, current_count).
        """
        results = self.model(
            frame,
            conf=self.confidence,
            classes=[COW_CLASS_ID],
            verbose=False,
            imgsz=640,
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        # ── Statistics ──────────────────────────────────────────────────────
        current_count = len(detections)
        self.stats.total_frames += 1
        self.stats.frame_counts.append(current_count)
        self.stats.max_simultaneous = max(self.stats.max_simultaneous, current_count)

        if detections.tracker_id is not None:
            for tid in detections.tracker_id:
                self.stats.unique_ids.add(int(tid))

        # ── Build per-detection labels ───────────────────────────────────────
        labels = []
        if detections.tracker_id is not None:
            for i, tid in enumerate(detections.tracker_id):
                conf = detections.confidence[i] if detections.confidence is not None else 0.0
                labels.append(f" Boi #{tid}  {conf:.0%} ")

        # ── Annotate frame ───────────────────────────────────────────────────
        annotated = frame.copy()
        if len(detections) > 0:
            annotated = self.trace_annotator.annotate(annotated, detections)
            annotated = self.box_annotator.annotate(annotated, detections)
            if labels:
                annotated = self.label_annotator.annotate(annotated, detections, labels)

        self._draw_hud(annotated, current_count)
        return annotated, current_count

    def _draw_hud(self, frame: np.ndarray, current_count: int):
        """Draw semi-transparent HUD overlay with live counters."""
        h, w = frame.shape[:2]
        box_w, box_h = 320, 120
        x1, y1 = 10, 10
        x2, y2 = x1 + box_w, y1 + box_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 35, 10), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 210, 50), 2)

        cv2.putText(frame, "B O V S M A R T", (x1 + 10, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 80), 2)
        cv2.line(frame, (x1 + 5, y1 + 38), (x2 - 5, y1 + 38), (50, 150, 50), 1)
        cv2.putText(frame, f"Bois na tela:    {current_count}",
                    (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        cv2.putText(frame, f"Total unicos:    {self.stats.total_unique}",
                    (x1 + 10, y1 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (100, 255, 100), 2)
