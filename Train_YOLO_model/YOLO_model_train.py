
if __name__ == '__main__':
    from ultralytics import YOLO
    model1 = YOLO('yolov8n.pt')

    model1.train(data="/kaggle/input/human-640/data.yaml",epochs=100,imgsz=640, batch=10)
