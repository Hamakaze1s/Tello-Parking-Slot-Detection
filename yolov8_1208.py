import argparse
import time
import os
import pyautogui
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch

def detect(opt):
    source, weights, conf_thres, iou_thres, device = (
        opt.source,
        opt.weights,
        opt.conf_thres,
        opt.iou_thres,
        opt.device,
    )

    # Set device
    device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load YOLOv8 model
    model = YOLO(weights)

    # Enable half precision if CUDA is available
    if device == 'cuda':
        model.model.half()  # Convert model to FP16

    # Main loop
    print("Starting detection...")
    path = 'data/test'
    while True:
        if __name__ != "__main__":
            pyautogui.press('s')
            path = os.getcwd()
        filelist = [f for f in os.listdir(path) if f.endswith((".png", ".jpg"))]

        for image in filelist:
            img_path = os.path.join(path, image)

            # Perform inference
            results = model.predict(source=img_path, conf=conf_thres, iou=iou_thres, device=device)

            # Process results
            img = cv2.imread(img_path)
            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy.cpu().numpy().flatten()  # 将二维数组展平为一维数组
                    x1, y1, x2, y2 = map(int, xyxy[:4])  # 确保只取前4个值

                    conf = box.conf.cpu().numpy().item()
                    cls = box.cls.cpu().numpy().item()

                    print(f"Detected {model.names[int(cls)]} with confidence {conf:.2f}")

                    # Draw bounding box
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    if model.names[int(cls)] != "empty":
                        continue
                    # x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display image
            cv2.imshow("YOLOv8 Detection", img)
            cv2.waitKey(10)
            if __name__ != "__main__":
                # Remove processed image
                os.remove(img_path)



def yolo_v8():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8s_1208.pt', help='Path to YOLOv8 model')
    parser.add_argument('--source', type=str, default='0', help='Source: webcam (0) or image directory')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cuda, or cpu')
    opt = parser.parse_args()

    detect(opt)


if __name__ == '__main__':
    yolo_v8()
