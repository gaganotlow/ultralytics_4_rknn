from ultralytics import YOLO
model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt') # build from YAML and transfer weights
# Train the model
results = model.train(data='./plate-pose.yaml', epochs=100, imgsz=640, workers=2, batch=8,project="gaga_pose", name="test")