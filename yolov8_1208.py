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
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load YOLOv8 model
    model = YOLO(weights)

    # Check if input is a webcam or image directory
    webcam = source.isnumeric()
    
    print("Starting detection")

    # Main loop
    while True:
        pyautogui.press('s')
        path = os.getcwd()
        filelist = os.listdir(path)

        for image in filelist:
            if image.endswith(".png"):
                source = image

                # Load image
                img_path = os.path.join(path, source)
                img = cv2.imread(img_path)

                # Move image to GPU if available
                img_tensor = torch.from_numpy(img).float().to(device)
                img_tensor /= 255.0  # Normalize image
                if img_tensor.ndimension() == 3:
                    img_tensor = img_tensor.unsqueeze(0)

                # Enable half precision if CUDA is available
                if device == 'cuda' and torch.cuda.is_available():
                    model.model.half()  # Convert model to FP16
                    img_tensor = img_tensor.half()

                # Perform inference
                results = model.predict(img_tensor, conf=conf_thres, iou=iou_thres, device=device, show=True)

                # Process results
                for result in results:
                    for box in result.boxes:
                        xyxy = box.xyxy.cpu().numpy()
                        conf = box.conf.cpu().numpy()
                        cls = box.cls.cpu().numpy()

                        print(f"Detected {model.names[int(cls)]} with confidence {conf:.2f}")

                        # Draw bounding box
                        label = f"{model.names[int(cls)]} {conf:.2f}"
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display image
                cv2.imshow("YOLOv8 Detection", img)
                cv2.waitKey(1)

                # Remove processed image
                os.remove(img_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8s.pt', help='Path to YOLOv8 model')
    parser.add_argument('--source', type=str, default='0', help='Source: webcam (0) or image directory')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cuda, or cpu')
    opt = parser.parse_args()

    detect(opt)


if __name__ == '__main__':
    main()
