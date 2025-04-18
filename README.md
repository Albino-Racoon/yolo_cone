# YOLO Cone Detection

Projekt za detekcijo stožcev z uporabo YOLOv8. Model je treniran za prepoznavo treh tipov stožcev:
- Modri stožci
- Oranžni stožci
- Rumeni stožci

## Nastavitev

1. Ustvari virtualno okolje:
```bash
python -m venv .venv
```

2. Aktiviraj virtualno okolje:
```bash
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Namesti potrebne knjižnice:
```bash
pip install -r requirements.txt
```

## Uporaba

### Treniranje modela
Za treniranje modela uporabi:
```bash
python yolo_superior.py
```

### Detekcija v realnem času
Za detekcijo stožcev preko spletne kamere:
```bash
python detect_webcam.py
```

## Datoteke
- `yolo_superior.py` - Skripta za treniranje modela z naprednimi nastavitvami
- `detect_webcam.py` - Skripta za detekcijo v realnem času
- `requirements.txt` - Seznam potrebnih Python knjižnic
