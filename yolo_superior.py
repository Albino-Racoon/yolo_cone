from ultralytics import YOLO
import os
import torch
from pathlib import Path
import shutil
import random
import numpy as np

def set_seed(seed=42):
    """Nastavi vse možne seed-e za deterministično delovanje"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def prepare_workspace(data_yaml_path):
    """Pripravi delovni prostor in preveri podatke"""
    # Ustvari mapo za rezultate če ne obstaja
    save_dir = Path('superior_results')
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)
    return save_dir

def train_superior_model():
    # Nastavi seed za ponovljivost
    set_seed(42)
    
    # Dobimo absolutno pot do trenutnega direktorija
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(current_dir, 'moj-dataset', 'data.yaml')
    
    # Pripravi delovni prostor
    save_dir = prepare_workspace(data_yaml_path)
    
    # Inicializacija večjega modela za boljše rezultate
    model = YOLO('yolov8m.pt')  # Uporabimo medium model za boljše rezultate
    
    # Nastavitev parametrov za treniranje
    results = model.train(
        data=data_yaml_path,
        epochs=300,    # Še več epoh za boljše učenje
        imgsz=640,    # Večja velikost slike za boljšo natančnost
        batch=4,      # Manjši batch zaradi CPU omejitev
        device='cpu',
        workers=4,
        project=str(save_dir),
        name='exp',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',    # Naprednejši optimizer
        lr0=0.001,    # Začetni learning rate
        lrf=0.0001,   # Končni learning rate
        momentum=0.937,
        weight_decay=0.0005,  # L2 regularizacija
        warmup_epochs=15,     # Počasno ogrevanje
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=True,    # Rectangular training za boljšo učinkovitost
        cos_lr=True,  # Cosine learning rate scheduling
        close_mosaic=30,      # Kasneje zapremo mosaic
        resume=False,
        amp=False,    # Brez mixed precision na CPU
        fraction=1.0,
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.2,  # Dropout za boljšo generalizacijo
        val=True,
        save=True,
        save_json=True,
        save_hybrid=True,
        conf=0.001,   # Nizek confidence threshold med treningom
        iou=0.7,      # Višji IoU threshold za boljšo natančnost
        max_det=300,
        half=False,
        dnn=False,
        plots=True,
        patience=100,  # Early stopping patience
        # Augmentacija podatkov
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,    # Saturation augmentation
        hsv_v=0.4,    # Value augmentation
        degrees=10.0,  # Rotacija do 10 stopinj
        translate=0.2, # Translacija do 20%
        scale=0.5,    # Skaliranje +/- 50%
        shear=2.0,    # Shear augmentation
        perspective=0.0001,  # Perspective augmentation
        flipud=0.5,   # Flip up-down
        fliplr=0.5,   # Flip left-right
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.1,    # Mixup augmentation
        copy_paste=0.1, # Copy-paste augmentation
    )
    
    # Izvozi model v različne formate
    print("\nIzvažam model v različne formate...")
    try:
        model.export(format='onnx', dynamic=True, simplify=True)  # ONNX format
        print("ONNX format uspešno izvožen")
    except Exception as e:
        print(f"Napaka pri izvozu ONNX: {e}")
    
    try:
        model.export(format='openvino', dynamic=True)  # OpenVINO format
        print("OpenVINO format uspešno izvožen")
    except Exception as e:
        print(f"Napaka pri izvozu OpenVINO: {e}")
    
    print("\nTreniranje končano! Rezultati:")
    print(f"Najboljši model: {save_dir}/exp/weights/best.pt")
    print(f"Zadnji model: {save_dir}/exp/weights/last.pt")
    print(f"Grafi in metrike: {save_dir}/exp")
    
    # Izpiši končne metrike
    metrics = results.results_dict
    print("\nKončne metrike:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    train_superior_model() 