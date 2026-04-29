from ultralytics import YOLO

# Load a model (YOLO26 is recommended for best results)
model = YOLO("yolo26n.pt")

# Train the model
results = model.train(
    data="datasets/adidas_puma_balanced/data.yaml", epochs=100, imgsz=640, device="mps"
)
