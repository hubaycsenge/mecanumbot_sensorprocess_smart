from ultralytics import YOLO

# Load a YOLO26 model
model = YOLO("./yolo26m-pose.pt")

# Export the model to ONNX format
model.export(format="onnx",imgsz=1280)  # creates 'yolo26m-pose.onnx'