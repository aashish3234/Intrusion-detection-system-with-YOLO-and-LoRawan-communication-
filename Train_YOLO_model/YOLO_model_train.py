
if __name__ == '__main__':
    from ultralytics import YOLO
    model1 = YOLO('yolov8n.pt')

    model1.train(data="/kaggle/input/aquarium-data-cots/aquarium_pretrain/data.yaml",epochs=100,imgsz=640, batch=10)
