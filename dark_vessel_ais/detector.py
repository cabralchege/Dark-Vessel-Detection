import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class VesselDetector:
    def __init__(self, model_path, conf_thresh=0.25, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = YOLO(model_path).to(device)
        self.sahi_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=conf_thresh,
            device=device
        )
        self.conf_thresh = conf_thresh

    def detect(self, image_path, use_sahi=True, slice_size=640):
        """
        Run detection on an image.
        Returns: boxes (list of [x1,y1,x2,y2]), scores, image_shape (h,w)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = image.shape[:2]
        
        if use_sahi:
            result = get_sliced_prediction(
                image,
                self.sahi_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )
            boxes = []
            scores = []
            for obj in result.object_prediction_list:
                boxes.append([obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy])
                scores.append(obj.score.value)
        else:
            results = self.model(image, verbose=False)[0]
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy().tolist()
                scores = results.boxes.conf.cpu().numpy().tolist()
            else:
                boxes, scores = [], []
        return boxes, scores, (h, w)