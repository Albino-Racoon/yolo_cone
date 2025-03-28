import cv2
from ultralytics import YOLO
import numpy as np
import time

def main():
    # Naloži YOLO model
    print("Nalagam YOLO model...")
    model = YOLO("runs/train/exp/weights/best.pt")
    
    # Odpri kamero (0 je običajno privzeta spletna kamera)
    print("Odpiranje kamere...")
    cap = cv2.VideoCapture(0)
    
    # Nastavi resolucijo kamere na 640x480 za boljšo performanco
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Preveri, če je kamera uspešno odprta
    if not cap.isOpened():
        print("Napaka: Ne morem odpreti kamere!")
        return
    
    print("Začenjam detekcijo... Pritisnite 'q' za izhod.")
    
    # Barve za različne razrede (RGB format)
    colors = {
        'blue_cone': (255, 0, 0),    # Modra
        'orange_cone': (0, 165, 255), # Oranžna
        'yellow_cone': (0, 255, 255)  # Rumena
    }
    
    while True:
        # Zajemi sliko iz kamere
        ret, frame = cap.read()
        if not ret:
            print("Napaka pri zajemu slike!")
            break
            
        # Začni merjenje časa
        start_time = time.time()
        
        # Izvedi detekcijo
        results = model(frame, conf=0.5)  # Nastavi confidence threshold na 0.5
        
        # Izračunaj FPS
        fps = 1.0 / (time.time() - start_time)
        
        # Izriši rezultate na sliko
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Dobi koordinate box-a
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Dobi razred in verjetnost
                cls = result.names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                # Izberi barvo za razred
                color = colors.get(cls, (0, 255, 0))  # Privzeto zelena, če razred ni definiran
                
                # Izriši box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Izriši oznako
                label = f'{cls} {conf:.2f}'
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, color, -1)  # Filled
                cv2.putText(frame, label, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Izriši FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Prikaži sliko
        cv2.imshow('YOLO Detekcija', frame)
        
        # Preveri za tipko 'q' za izhod
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Počisti
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 