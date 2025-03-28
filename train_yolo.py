from ultralytics import YOLO
import os

def train_model():
    # Dobimo absolutno pot do trenutnega direktorija
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(current_dir, 'moj-dataset', 'data.yaml')
    
    # Inicializacija YOLOv8 modela
    model = YOLO('yolov8n.pt')  # Uporabimo najmanjši model za hitrejše treniranje
    
    # Nastavitev parametrov za treniranje
    results = model.train(
        data=data_yaml_path,  # Absolutna pot do data.yaml
        epochs=50,     # Zmanjšano število epoh
        imgsz=416,    # Manjša velikost slike
        batch=8,      # Manjši batch size za CPU
        device='cpu', # Uporaba CPU
        workers=4,    # Zmanjšano število delavcev
        project=os.path.join(current_dir, 'runs/train'),  # Absolutna pot do mape za rezultate
        name='exp',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=False,    # Izključena mixed precision training za CPU
        fraction=1.0,
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        save=True,
        save_json=False,
        save_hybrid=False,
        conf=0.001,
        iou=0.6,
        max_det=300,
        half=False,
        dnn=False,
        plots=True
    )
    
    # Shrani model
    model.export(format='onnx')
    print("Treniranje končano! Model je shranjen v mapi 'runs/train/exp'")

if __name__ == "__main__":
    train_model() 