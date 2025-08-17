from ultralytics import YOLO


if __name__ == "__main__":
    # Load a model
    model = YOLO(r"/content/drive/MyDrive/Steel_surface_defect_detection/runs/detect/train14/weights/best.pt")
    data="ultralytics/cfg/datasets/mydata.yaml"  # path to dataset YAML

    # Train the model
    train_results = model.val(
        data = data,
        imgsz=640,           # training image size
        device="cuda",         # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch = 10,
        workers = 24,
        split = "val"
    )
