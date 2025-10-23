from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("/home/ga/code/ultralytics-pose/gaga_pose/test2/weights/best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("/home/ga/code/ultralytics-pose/datasets/ceshi/images/train", save=True, imgsz=640, conf=0.50, iou=0.45, device=[0]) 
