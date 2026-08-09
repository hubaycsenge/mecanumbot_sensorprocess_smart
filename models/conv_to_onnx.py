from ultralytics import YOLO

# Load a YOLO26 model
model = YOLO("./yolo26s-pose.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo26m-pose.onnx'