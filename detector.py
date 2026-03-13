import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Set, List, Optional

# COCO class ID for "cow" — used when model is a standard YOLO/COCO model
COCO_COW_CLASS_ID = 19

MODELS = {
    "Nano — Mais Rápido": "yolov8n.pt",
    "Small — Rápido": "yolov8s.pt",
    "Medium — Recomendado ✓": "yolov8m.pt",
    "Large — Alta Precisão": "yolov8l.pt",
    "XLarge — Máxima Precisão": "yolov8x.pt",
}

PALETTE_HEX = [
    "#FF6B35", "#F7C948", "#4CAF50", "#2196F3",
    "#9C27B0", "#F44336", "#00BCD4", "#FF5722",
    "#8BC34A", "#E91E63", "#00E676", "#FFEB3B",
    "#FF4081", "#69F0AE", "#40C4FF", "#EA80FC",
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
    def __init__(
        self,
        model_key: str = "Medium — Recomendado ✓",
        custom_model_path: Optional[str] = None,
        confidence: float = 0.25,
        iou: float = 0.30,
        drone_mode: bool = True,
        tile_size: int = 640,
        tile_overlap: float = 0.20,
        imgsz: int = 1280,
        cow_class_id: int = COCO_COW_CLASS_ID,
        max_inference_size: int = 1280,   # resize frame before SAHI (key speed opt)
        perform_standard_pred: bool = False,  # full-frame pass on top of tiles
    ):
        self.confidence = confidence
        self.iou = iou
        self.drone_mode = drone_mode
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.imgsz = imgsz
        self.cow_class_id = cow_class_id
        self.max_inference_size = max_inference_size
        self.perform_standard_pred = perform_standard_pred

        # ── Load model ──────────────────────────────────────────────────────
        if custom_model_path:
            model_path = custom_model_path
        else:
            model_path = MODELS.get(model_key, "yolov8m.pt")

        self.model = YOLO(model_path)

        # ── SAHI model (lazy-loaded on first drone frame) ────────────────────
        self._sahi_model = None
        self._sahi_model_path = model_path

        self.stats = CattleStats()
        self._init_tracker()
        self._init_annotators()

    def _get_sahi_model(self):
        """Lazy-load SAHI model to avoid import overhead when not needed."""
        if self._sahi_model is None:
            import torch
            from sahi import AutoDetectionModel
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=self._sahi_model_path,
                confidence_threshold=self.confidence,
                device=device,
            )
        return self._sahi_model

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
            thickness=2,
        )
        self.label_annotator = sv.LabelAnnotator(
            color=palette,
            color_lookup=lookup,
            text_color=sv.Color.WHITE,
            text_scale=0.55,
            text_thickness=1,
            text_padding=4,
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=palette,
            color_lookup=lookup,
            thickness=2,
            trace_length=80,
        )

    def reset(self):
        self.stats = CattleStats()
        self._init_tracker()

    # ── Public entry point ───────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> tuple:
        """Run detection + tracking. Returns (annotated_frame, current_count)."""
        if self.drone_mode:
            # Resize for inference, detect, scale boxes back to original size
            small, scale = self._resize_for_inference(frame)
            detections = self._detect_sahi(small)
            if scale != 1.0 and len(detections) > 0:
                detections.xyxy = detections.xyxy / scale
        else:
            detections = self._detect_standard(frame)

        detections = self.tracker.update_with_detections(detections)

        # ── Statistics ───────────────────────────────────────────────────────
        current_count = len(detections)
        self.stats.total_frames += 1
        self.stats.frame_counts.append(current_count)
        self.stats.max_simultaneous = max(self.stats.max_simultaneous, current_count)

        if detections.tracker_id is not None:
            for tid in detections.tracker_id:
                self.stats.unique_ids.add(int(tid))

        # ── Labels ───────────────────────────────────────────────────────────
        labels = []
        if detections.tracker_id is not None:
            for i, tid in enumerate(detections.tracker_id):
                conf = detections.confidence[i] if detections.confidence is not None else 0.0
                labels.append(f" #{tid} {conf:.0%} ")

        # ── Annotate ─────────────────────────────────────────────────────────
        annotated = frame.copy()
        if len(detections) > 0:
            annotated = self.trace_annotator.annotate(annotated, detections)
            annotated = self.box_annotator.annotate(annotated, detections)
            if labels:
                annotated = self.label_annotator.annotate(annotated, detections, labels)

        self._draw_hud(annotated, current_count)
        return annotated, current_count

    # ── Resize helper ────────────────────────────────────────────────────────

    def _resize_for_inference(self, frame: np.ndarray) -> tuple:
        """
        Downscale frame so its longest side ≤ max_inference_size.
        Returns (resized_frame, scale_factor).
        scale_factor < 1 means the frame was shrunk.
        Example: 4K (3840x2160) → max 1280 → scale=0.333 → ~9x fewer SAHI tiles.
        """
        h, w = frame.shape[:2]
        max_side = self.max_inference_size
        if max(h, w) <= max_side:
            return frame, 1.0
        scale = max_side / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    # ── Detection backends ───────────────────────────────────────────────────

    def _detect_standard(self, frame: np.ndarray) -> sv.Detections:
        """Standard single-pass YOLO inference (ground-level footage)."""
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.cow_class_id],
            verbose=False,
            imgsz=self.imgsz,
        )[0]
        return sv.Detections.from_ultralytics(results)

    def _detect_sahi(self, frame: np.ndarray) -> sv.Detections:
        """
        SAHI sliced inference for drone/aerial footage.
        Splits the high-res frame into overlapping tiles, runs YOLO on each,
        then merges results — critical for detecting small animals from altitude.
        """
        from sahi.predict import get_sliced_prediction
        import PIL.Image

        pil_image = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        result = get_sliced_prediction(
            pil_image,
            self._get_sahi_model(),
            slice_height=self.tile_size,
            slice_width=self.tile_size,
            overlap_height_ratio=self.tile_overlap,
            overlap_width_ratio=self.tile_overlap,
            perform_standard_pred=self.perform_standard_pred,
            postprocess_type="GREEDYNMM",
            postprocess_match_threshold=self.iou,
            verbose=0,
        )

        return self._sahi_to_sv(result)

    def _sahi_to_sv(self, result) -> sv.Detections:
        """Convert SAHI ObjectPredictionList → supervision Detections."""
        preds = result.object_prediction_list

        # Filter to cow class only (for COCO models)
        preds = [p for p in preds if p.category.id == self.cow_class_id]

        if not preds:
            return sv.Detections.empty()

        boxes = np.array(
            [[p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy] for p in preds],
            dtype=np.float32,
        )
        scores = np.array([p.score.value for p in preds], dtype=np.float32)
        class_ids = np.array([p.category.id for p in preds], dtype=int)

        return sv.Detections(xyxy=boxes, confidence=scores, class_id=class_ids)

    # ── HUD overlay ──────────────────────────────────────────────────────────

    def _draw_hud(self, frame: np.ndarray, current_count: int):
        h, w = frame.shape[:2]
        # Scale HUD with frame size so it's readable on 4K
        scale = max(1.0, w / 1920)
        box_w = int(330 * scale)
        box_h = int(130 * scale)
        x1, y1 = 10, 10
        x2, y2 = x1 + box_w, y1 + box_h
        fs = 0.65 * scale
        th = max(1, int(2 * scale))
        lh = int(35 * scale)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 35, 10), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 210, 50), 2)

        mode_tag = "DRONE" if self.drone_mode else "SOLO"
        cv2.putText(frame, f"BOVSMART  [{mode_tag}]", (x1 + 10, y1 + lh),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (80, 255, 80), th)
        cv2.line(frame, (x1 + 5, y1 + lh + 8), (x2 - 5, y1 + lh + 8), (50, 150, 50), 1)
        cv2.putText(frame, f"Bois na tela:    {current_count}",
                    (x1 + 10, y1 + lh * 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)
        cv2.putText(frame, f"Total unicos:    {self.stats.total_unique}",
                    (x1 + 10, y1 + lh * 3 + 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (100, 255, 100), th)
